"""Endpoints de la API.

Aqui van todos los endpoints que el frontend (navegador) llama, bajo el
prefijo /api (ver main.py). Todo lo relacionado a chats se guarda en
SQLite (chat_store.py) y las respuestas las genera el motor RAG real
(rag_service.py, que envuelve a Agentes/rag_agent.py).

AUTENTICACION: todos los endpoints de chats exigen un usuario logueado
(Depends(get_current_user)) y operan SOLO sobre los chats de ese usuario.
El chequeo de propiedad es automatico: chat_store.obtener_chat(chat_id,
usuario_id) devuelve None si el chat no es del usuario, lo que se traduce
en un 404 — un usuario nunca ve ni toca los chats de otro. Los endpoints
/auth/* (login, logout) y /health son publicos.

AZURE: la dependencia get_current_user (auth.py) es la costura que en
Fase 2 se reimplementa contra Entra ID sin tocar ni un endpoint de aqui.
"""

import json
import os
import traceback

from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chat_store
import rag_service
import auth

router = APIRouter()

# Cookie segura (solo por HTTPS) en produccion; en local (http) debe ir en
# false o el navegador no la guarda. AZURE: poner IAPY_COOKIE_SECURE=true.
_COOKIE_SECURE = os.getenv("IAPY_COOKIE_SECURE", "false").lower() == "true"


# ==========================================================================
# MODELOS (lo que el frontend envia)
# ==========================================================================

class MensajeEntrante(BaseModel):
    """Lo que manda el frontend al escribir en el chat."""
    texto: str


class NuevoChat(BaseModel):
    titulo: str = "Nuevo chat"


class TituloChat(BaseModel):
    """Para renombrar un chat existente."""
    titulo: str


class CredencialesLogin(BaseModel):
    email: str
    password: str


class NuevoUsuario(BaseModel):
    email: str
    nombre: str
    password: str
    rol: str = "usuario"


class CambioRol(BaseModel):
    rol: str


# ==========================================================================
# AUTENTICACION  (publico)
# ==========================================================================

@router.post("/auth/login")
async def login(credenciales: CredencialesLogin, response: Response):
    """Verifica email + contraseña y, si son correctos, abre la sesion."""
    usuario = chat_store.obtener_usuario_por_email(credenciales.email)
    # Se verifica el hash SIEMPRE (aunque el usuario no exista) para no
    # revelar por tiempo de respuesta si un email esta registrado o no.
    hash_guardado = usuario["hash_password"] if usuario else "pbkdf2_sha256$1$00$00"
    if not auth.verificar_password(credenciales.password, hash_guardado) or not usuario:
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")

    token = auth.crear_token_sesion(usuario["id"])
    response.set_cookie(
        key=auth.NOMBRE_COOKIE,
        value=token,
        max_age=auth.DURACION_SESION_SEG,
        httponly=True,          # inaccesible desde JavaScript (anti-XSS)
        samesite="lax",         # no viaja en peticiones cross-site (anti-CSRF)
        secure=_COOKIE_SECURE,
    )
    return {"id": usuario["id"], "nombre": usuario["nombre"], "rol": usuario["rol"]}


@router.post("/auth/logout")
async def logout(response: Response):
    """Cierra la sesion borrando la cookie."""
    response.delete_cookie(key=auth.NOMBRE_COOKIE)
    return {"ok": True}


@router.get("/auth/me")
async def quien_soy(usuario: dict = Depends(auth.get_current_user)):
    """Datos del usuario logueado (lo usa el frontend al cargar)."""
    return {"id": usuario["id"], "nombre": usuario["nombre"],
            "email": usuario["email"], "rol": usuario["rol"]}


# ==========================================================================
# SALUD  (publico)
# ==========================================================================

@router.get("/health")
async def health():
    """Chequeo simple: si responde, el servidor esta vivo."""
    return {"estado": "ok"}


# ==========================================================================
# CHATS  (requieren sesion; acotados al usuario dueño)
# ==========================================================================

@router.get("/chats")
async def listar_chats(usuario: dict = Depends(auth.get_current_user)):
    """Los chats del usuario logueado, mas reciente primero."""
    return chat_store.listar_chats(usuario["id"])


@router.post("/chats")
async def crear_chat(datos: NuevoChat, usuario: dict = Depends(auth.get_current_user)):
    """Crea un chat nuevo (vacio) para el usuario y lo devuelve."""
    return chat_store.crear_chat(usuario["id"], datos.titulo)


@router.get("/chats/{chat_id}/mensajes")
async def obtener_mensajes(chat_id: str, usuario: dict = Depends(auth.get_current_user)):
    """Historial completo de un chat del usuario, para pintarlo al abrirlo."""
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    return chat_store.obtener_mensajes(chat_id)


@router.patch("/chats/{chat_id}")
async def renombrar_chat(chat_id: str, datos: TituloChat,
                         usuario: dict = Depends(auth.get_current_user)):
    """Cambia el titulo de un chat del usuario (renombrado manual)."""
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    titulo = datos.titulo.strip()
    if not titulo:
        raise HTTPException(status_code=400, detail="El titulo esta vacio")
    chat_store.renombrar_chat(chat_id, titulo[:100])
    return chat_store.obtener_chat(chat_id, usuario["id"])


@router.delete("/chats/{chat_id}")
async def eliminar_chat(chat_id: str, usuario: dict = Depends(auth.get_current_user)):
    """Borra un chat del usuario y todos sus mensajes."""
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    chat_store.eliminar_chat(chat_id)
    return {"eliminado": True}


@router.delete("/chats/{chat_id}/ultimo-turno")
async def eliminar_ultimo_turno(chat_id: str, usuario: dict = Depends(auth.get_current_user)):
    """Borra el ultimo turno (ultima pregunta del usuario + su respuesta).

    Lo usa la funcion "editar y reenviar" de la GUI: primero se borra el
    turno viejo con este endpoint, y luego el frontend manda el texto
    editado por el endpoint de streaming normal, como un mensaje nuevo.
    """
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
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

    Nota: la propiedad del chat ya la verifico el endpoint que llama aqui.
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
def enviar_mensaje(chat_id: str, mensaje: MensajeEntrante,
                   usuario: dict = Depends(auth.get_current_user)):
    """Recibe una pregunta, la pasa por el RAG y guarda ambos lados de la charla.

    Nota: esta funcion NO es 'async def' a proposito. FastAPI ejecuta las
    funciones sincronas ('def' normal) en un threadpool automaticamente,
    lo cual es justo lo que queremos aqui porque rag_service.responder()
    es una llamada bloqueante de larga duracion (Ollama + ChromaDB).
    """
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
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
def enviar_mensaje_stream(chat_id: str, mensaje: MensajeEntrante,
                          usuario: dict = Depends(auth.get_current_user)):
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
    if not chat_store.obtener_chat(chat_id, usuario["id"]):
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


# ==========================================================================
# ADMINISTRACION  (requieren rol admin)
# ==========================================================================
# El admin tiene todos los permisos: gestionar usuarios (crear, listar,
# cambiar rol, eliminar). No puede leer los chats de otros usuarios por
# diseño — la privacidad del historial se respeta incluso para el admin;
# lo que administra son cuentas, no conversaciones ajenas.

@router.get("/admin/usuarios")
async def admin_listar_usuarios(admin: dict = Depends(auth.require_admin)):
    """Lista todos los usuarios (sin hashes de contraseña)."""
    return chat_store.listar_usuarios()


@router.post("/admin/usuarios")
async def admin_crear_usuario(datos: NuevoUsuario, admin: dict = Depends(auth.require_admin)):
    """Crea un usuario nuevo con su contraseña hasheada."""
    if datos.rol not in ("usuario", "admin"):
        raise HTTPException(status_code=400, detail="Rol invalido (usa 'usuario' o 'admin')")
    if len(datos.password) < 8:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres")
    try:
        return chat_store.crear_usuario(
            datos.email, datos.nombre, auth.hash_password(datos.password), datos.rol
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.patch("/admin/usuarios/{user_id}/rol")
async def admin_cambiar_rol(user_id: str, datos: CambioRol,
                            admin: dict = Depends(auth.require_admin)):
    """Cambia el rol de un usuario (usuario <-> admin)."""
    if datos.rol not in ("usuario", "admin"):
        raise HTTPException(status_code=400, detail="Rol invalido")
    if not chat_store.obtener_usuario_por_id(user_id):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    chat_store.cambiar_rol(user_id, datos.rol)
    return chat_store.obtener_usuario_por_id(user_id)


@router.delete("/admin/usuarios/{user_id}")
async def admin_eliminar_usuario(user_id: str, admin: dict = Depends(auth.require_admin)):
    """Elimina un usuario y todos sus chats. No puede eliminarse a si mismo."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="No puedes eliminar tu propia cuenta")
    if not chat_store.obtener_usuario_por_id(user_id):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    chat_store.eliminar_usuario(user_id)
    return {"eliminado": True}
