"""Puente entre la GUI web y el motor RAG real (Agentes/rag_agent.py).

Este modulo se importa una sola vez cuando arranca el servidor (ver main.py),
y en ese momento construye los componentes "caros" del RAG (indice BM25,
cliente de ChromaDB, modelo de embeddings) UNA sola vez. Cada pregunta
reutiliza esas mismas instancias; lo unico que se crea por conversacion es
la memoria de la charla, reconstruida desde el historial guardado en SQLite.
"""

import sys
from pathlib import Path
from typing import Dict, Iterator, List

# Agentes/ vive al lado de web/. Lo agregamos al sys.path para poder
# reutilizar las clases de rag_agent.py sin duplicar codigo.
# AZURE (Fase 1): este hack desaparece al empaquetar el proyecto — con
# pyproject.toml el import pasa a ser "from iapy_rag.agents import ...".
AGENTES_DIR = Path(__file__).resolve().parent.parent / "Agentes"
if str(AGENTES_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTES_DIR))

# rag_agent.py llama a dotenv.load_dotenv() sin ruta, que busca el .env
# subiendo desde el directorio de trabajo actual. Como web/ es HERMANO de
# Agentes/ (no un padre), esa busqueda nunca encontraria Agentes/.env si el
# servidor se lanza desde web/. Lo cargamos aqui primero, con ruta explicita,
# para que las variables (DD_API_KEY, DD_LLMOBS_ML_APP) ya esten en el
# entorno antes de que rag_agent.py haga su propio load_dotenv() (que en ese
# punto sera un no-op inofensivo porque las variables ya existen).
from dotenv import load_dotenv
load_dotenv(AGENTES_DIR / ".env")

from rag_agent import (  # noqa: E402  (import despues de tocar sys.path a proposito)
    RAGConfig,
    VectorSearchEngine,
    LexicalSearchEngine,
    ParentStore,
    HybridRetrievalService,
    SlidingWindowMemory,
    MemoryAwareTranslationAgent,
    SynthesisAgent,
    RAGOrchestrator,
    CrossEncoderReranker,
)

print("[rag_service] Inicializando motor RAG (esto puede tardar un poco la primera vez)...")

# Rutas absolutas hacia la base vectorial dentro de Agentes/, sin importar
# desde donde se lance uvicorn.
_config = RAGConfig(
    db_path=AGENTES_DIR / "mi_base_vectorial",
    parent_store_path=AGENTES_DIR / "mi_base_vectorial" / "parent_store.json",
)

_vector_engine = VectorSearchEngine(_config)

_corpus = _vector_engine.fetch_all_data()
_lexical_engine = LexicalSearchEngine(_corpus["documents"], _corpus["metadatas"])

_parent_store = ParentStore(_config.parent_store_path)
_hybrid_service = HybridRetrievalService(_config, _vector_engine, _lexical_engine)

_expander = MemoryAwareTranslationAgent(_config.llm_model)
_synthesizer = SynthesisAgent(_config.llm_model)

# Re-ranking (cross-encoder), segunda etapa de precision. Se carga una sola
# vez aqui (igual que los demas componentes "caros") porque instanciar el
# modelo tiene costo; despues cada pregunta solo llama a .rerank().
_reranker = CrossEncoderReranker(_config.rerank_model) if _config.rerank_enabled else None

print("[rag_service] Motor RAG listo.")


def _construir_orquestador(historial_previo: List[Dict[str, str]]) -> RAGOrchestrator:
    """Arma un orquestador con la memoria reconstruida desde el historial.

    Los componentes caros (indices, modelos) son los compartidos del modulo;
    lo unico nuevo por conversacion es la memoria.
    """
    memoria = SlidingWindowMemory(_config.memory_window_size)
    for turno in historial_previo:
        if turno["rol"] == "user":
            memoria.add_user_message(turno["contenido"])
        else:
            memoria.add_ai_message(turno["contenido"])

    return RAGOrchestrator(
        config=_config,
        memory=memoria,
        retriever=_hybrid_service,
        expander=_expander,
        synthesizer=_synthesizer,
        parent_store=_parent_store,
        reranker=_reranker,
    )


def responder(pregunta: str, historial_previo: List[Dict[str, str]]) -> str:
    """Genera una respuesta completa (bloqueante) para `pregunta`.

    historial_previo: lista de dicts [{"rol": "user"|"assistant", "contenido": "..."}]
    en orden cronologico, tal como se recuperan de chat_store.py.
    """
    return _construir_orquestador(historial_previo).process_query(pregunta)


def responder_stream(pregunta: str, historial_previo: List[Dict[str, str]]) -> Iterator[str]:
    """Variante streaming: cede la respuesta por fragmentos a medida que el
    LLM la produce. La consume el endpoint SSE de api.py para que el
    navegador pinte la respuesta en vivo.
    """
    return _construir_orquestador(historial_previo).process_query_stream(pregunta)
