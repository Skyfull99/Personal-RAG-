"""Endpoints de la API.

Aqui van todos los endpoints que el frontend (navegador) llama, bajo el
prefijo /api (ver main.py). Todo lo relacionado a chats se guarda en
SQLite (chat_store.py) y las respuestas las genera el motor RAG real
(rag_service.py, que envuelve a Agentes/rag_agent.py).
"""

import json
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chat_store
import rag_service

router = APIRouter()


class MensajeEntrante(BaseModel):
    """Lo que manda el frontend al escribir en el chat."""
    texto: str


class NuevoChat(BaseModel):
    titulo: str = "Nuevo chat"


class TituloChat(BaseModel):
    """Para renombrar un chat existente."""
    titulo: str


@router.get("/health")
async def health():
    """Chequeo simple: si responde, el servidor esta vivo."""
    return {"estado": "ok"}


@router.get("/chats")
async def listar_chats():
    """Lista de todos los chats guardados, mas reciente primero."""
    return chat_store.listar_chats()


@router.post("/chats")
async def crear_chat(datos: NuevoChat):
    """Crea un chat nuevo (vacio) y lo devuelve."""
    return chat_store.crear_chat(datos.titulo)


@router.get("/chats/{chat_id}/mensajes")
async def obtener_mensajes(chat_id: str):
    """Historial completo de un chat, para pintarlo al abrirlo."""
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    return chat_store.obtener_mensajes(chat_id)


@router.patch("/chats/{chat_id}")
async def renombrar_chat(chat_id: str, datos: TituloChat):
    """Cambia el titulo de un chat (renombrado manual desde la GUI)."""
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    titulo = datos.titulo.strip()
    if not titulo:
        raise HTTPException(status_code=400, detail="El titulo esta vacio")
    chat_store.renombrar_chat(chat_id, titulo[:100])
    return chat_store.obtener_chat(chat_id)


@router.delete("/chats/{chat_id}")
async def eliminar_chat(chat_id: str):
    """Borra un chat y todos sus mensajes."""
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    chat_store.eliminar_chat(chat_id)
    return {"eliminado": True}


@router.delete("/chats/{chat_id}/ultimo-turno")
async def eliminar_ultimo_turno(chat_id: str):
    """Borra el ultimo turno (ultima pregunta del usuario + su respuesta).

    Lo usa la funcion "editar y reenviar" de la GUI: primero se borra el
    turno viejo con este endpoint, y luego el frontend manda el texto
    editado por el endpoint de streaming normal, como un mensaje nuevo.
    """
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    if not chat_store.eliminar_ultimo_turno(chat_id):
        raise HTTPException(status_code=404, detail="El chat no tiene mensajes para editar")
    return {"eliminado": True}


def _registrar_pregunta(chat_id: str, texto: str) -> list:
    """Lee el historial previo y guarda el mensaje del usuario DE INMEDIATO.

    Guardar la pregunta al recibirla (y no despues de generar la respuesta,
    como se hacia antes) resuelve dos problemas: si el RAG falla, la
    pregunta ya no se pierde del historial; y dos requests simultaneos al
    mismo chat ya no interlacean turnos (cada pregunta queda registrada en
    el orden en que llego). El historial que se devuelve es el PREVIO a
    esta pregunta: es lo que se pasa como memoria al RAG (la pregunta
    actual viaja aparte, no debe ir duplicada en la memoria).
    """
    historial_previo = chat_store.obtener_mensajes(chat_id)

    chat_store.agregar_mensaje(chat_id, "user", texto)

    # Si es el primer mensaje del chat, usarlo (recortado) como titulo.
    if not historial_previo:
        titulo = texto[:60] + ("..." if len(texto) > 60 else "")
        chat_store.renombrar_chat(chat_id, titulo)

    return [{"rol": m["rol"], "contenido": m["contenido"]} for m in historial_previo]


def _funcion_bloqueante_preguntar(chat_id: str, texto: str) -> str:
    """Se ejecuta en un hilo aparte (ver endpoint de abajo) porque el RAG
    hace llamadas sincronas al LLM local que pueden tardar decenas de
    segundos; asi no se bloquea el servidor para otros usuarios/pestañas.
    """
    historial_para_memoria = _registrar_pregunta(chat_id, texto)

    respuesta = rag_service.responder(texto, historial_para_memoria)

    chat_store.agregar_mensaje(chat_id, "assistant", respuesta)

    return respuesta


@router.post("/chats/{chat_id}/mensajes")
def enviar_mensaje(chat_id: str, mensaje: MensajeEntrante):
    """Recibe una pregunta, la pasa por el RAG y guarda ambos lados de la charla.

    Nota: esta funcion NO es 'async def' a proposito. FastAPI ejecuta las
    funciones sincronas ('def' normal) en un threadpool automaticamente,
    lo cual es justo lo que queremos aqui porque rag_service.responder()
    es una llamada bloqueante de larga duracion (Ollama + ChromaDB).
    """
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")

    texto = mensaje.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El mensaje esta vacio")

    try:
        respuesta = _funcion_bloqueante_preguntar(chat_id, texto)
    except ConnectionError:
        # Es el error exacto que lanza el cliente de ollama cuando el
        # servidor de Ollama no esta corriendo. Mensaje claro en vez de
        # un 500 con traceback crudo en la cara del usuario.
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar con Ollama. Verifica que este corriendo "
                   "(abre la app de Ollama o corre 'ollama serve') e intenta de nuevo.",
        )
    except Exception as e:
        # Cualquier otro fallo inesperado (ChromaDB, modelo no encontrado,
        # etc.): se registra completo en la consola del servidor para
        # depurar, pero al usuario se le manda un mensaje corto y claro.
        print("[ERROR] Fallo inesperado al procesar la pregunta:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado del RAG: {e}",
        )

    return {"respuesta": respuesta}


def _evento_sse(datos: dict) -> str:
    """Formatea un evento Server-Sent Events (una linea data: + doble salto)."""
    return f"data: {json.dumps(datos, ensure_ascii=False)}\n\n"


@router.post("/chats/{chat_id}/mensajes/stream")
def enviar_mensaje_stream(chat_id: str, mensaje: MensajeEntrante):
    """Version streaming del endpoint de mensajes (la que usa la GUI).

    Devuelve Server-Sent Events: cada fragmento de la respuesta del LLM
    viaja como {"delta": "..."} en cuanto se genera, asi el navegador
    pinta la respuesta en vivo en vez de esperarla completa. Al final va
    {"fin": true}; si algo falla a mitad, {"error": "..."} (a esa altura
    ya se enviaron los headers 200, no se puede cambiar el status code).

    La pregunta del usuario se guarda en SQLite al recibirla (ver
    _registrar_pregunta); la respuesta, solo cuando el stream se completo.
    Si el RAG falla o el cliente corta a mitad, queda la pregunta sin
    respuesta en el historial — que es lo honesto — en vez de perderse
    el turno entero como pasaba antes.

    Nota: la funcion y el generador son sincronos a proposito — Starlette
    itera generadores sincronos en su threadpool, que es lo que queremos
    porque el RAG (Ollama + ChromaDB) es bloqueante y lento.
    """
    if not chat_store.obtener_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat no encontrado")

    texto = mensaje.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El mensaje esta vacio")

    historial_para_memoria = _registrar_pregunta(chat_id, texto)

    def eventos():
        respuesta_completa = ""
        try:
            for fragmento in rag_service.responder_stream(texto, historial_para_memoria):
                respuesta_completa += fragmento
                yield _evento_sse({"delta": fragmento})
        except ConnectionError:
            yield _evento_sse({
                "error": "No se pudo conectar con Ollama. Verifica que este corriendo "
                         "(abre la app de Ollama o corre 'ollama serve') e intenta de nuevo."
            })
            return
        except Exception as e:
            print("[ERROR] Fallo inesperado al procesar la pregunta (stream):")
            traceback.print_exc()
            yield _evento_sse({"error": f"Error inesperado del RAG: {e}"})
            return

        chat_store.agregar_mensaje(chat_id, "assistant", respuesta_completa)

        yield _evento_sse({"fin": True})

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        # Evita que proxies/middleware intermedios acumulen el stream en
        # un buffer y lo entreguen de golpe (anularia el efecto en vivo).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
