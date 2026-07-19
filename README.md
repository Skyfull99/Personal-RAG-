# IAPY — Sistema RAG Local

Sistema de **Retrieval-Augmented Generation** 100% local (Ollama, sin nube): convierte documentos de la empresa y transcripciones de videos en una base de conocimiento consultable por chat, con citas de fuentes.

```
Documentos (.md) → Ingesta incremental (Chroma + Contextual Retrieval)
→ Búsqueda híbrida (vectorial + BM25, fusión RRF) → Re-ranking (cross-encoder + umbral)
→ Parent-Child Retrieval → Respuesta con streaming y citas
```

## Estructura del proyecto

| Carpeta | Qué es |
|---|---|
| `Agentes/` | **El motor RAG.** `ingest_markdown.py` (ingesta incremental) y `rag_agent.py` (consulta). Incluye su venv (`env/`) y la base vectorial (`mi_base_vectorial/`) — ambos fuera de Git. |
| `web/` | **Interfaz web** (FastAPI + frontend). Chats persistentes en SQLite, streaming SSE, edición de prompts, temas claro/oscuro. |
| `datos/fuentes/archivos-alta/` | Fuente de conocimiento **activa**: documentos de la empresa (PDF + sus versiones `_Markdown.md`). Fuera de Git. |
| `datos/fuentes/youtube-markdown/` | Fuente de conocimiento: ~48 transcripciones de videos ya formateadas en Markdown (hoy comentada en la config de ingesta). Fuera de Git. |
| `docs/` | Documentación: mapa histórico, planes de migración a Azure (Fases 1–3) y diagramas de arquitectura. |

> Los pipelines de producción de contenido (transcripción de YouTube y grabación de clases) se retiraron del proyecto — existen en el respaldo completo de la carpeta original. El RAG solo necesita los `.md` resultantes, que viven en `datos/fuentes/`.

## Cómo se usa

Todo corre con el venv de `Agentes/` y requiere **Ollama** activo (`ollama serve`) con los modelos `gemma4:e4b` y `nomic-embed-text`.

**1. Ingestar documentos** (incremental: solo procesa lo nuevo o modificado):

```powershell
cd Agentes
.\env\Scripts\python.exe ingest_markdown.py            # ingesta normal
.\env\Scripts\python.exe ingest_markdown.py --prune    # además elimina de la base los archivos borrados
.\env\Scripts\python.exe ingest_markdown.py --rebuild  # regenera TODO desde cero
```

Las carpetas fuente se configuran en `IngestionConfig.source_folders` (`ingest_markdown.py`); las rutas están ancladas a la ubicación del archivo, así que funcionan en cualquier máquina. El registro de lo ya ingestado vive en `mi_base_vectorial/ingest_manifest.json`.

**2. Consultar por consola:**

```powershell
cd Agentes
.\env\Scripts\python.exe rag_agent.py
```

**3. Interfaz web** (la forma normal de uso):

```powershell
cd web
..\Agentes\env\Scripts\python.exe -m uvicorn main:app --reload
# abrir http://localhost:8000
```

## Características del motor

- **Ingesta incremental y no destructiva**: manifest con hash + fecha por archivo; un corte a mitad no pierde nada; `--prune` y `--rebuild` explícitos.
- **Contextual Retrieval** (Anthropic): cada chunk se vectoriza enriquecido con keywords globales (muestreadas de inicio/medio/fin del documento) y contexto situacional generado por LLM.
- **Parent-Child Retrieval**: se buscan chunks precisos, se responde con la sección completa.
- **Búsqueda híbrida con RRF**: vectorial (coseno) + BM25 (con plegado de tildes para español), fusionadas por ranking — sin pesos que tunear.
- **Re-ranking con cross-encoder** (`bge-reranker-v2-m3`) sobre el contenido crudo del chunk, con **umbral de honestidad**: si nada es relevante, el sistema dice que no encontró información en vez de inventar.
- **Expansión de consultas** multilingüe con memoria de conversación; la query original siempre se busca literal.
- **Defensa contra inyección de prompt indirecta** en el sintetizador (los documentos son datos, nunca instrucciones).
- **Streaming de punta a punta**: la respuesta se pinta en vivo en el navegador (SSE).
- Observabilidad con **Datadog LLM Observability** (credenciales en `Agentes/.env`, fuera de Git).

## Qué versiona Git (y qué no)

El `.gitignore` excluye deliberadamente:

- **Secretos**: `Agentes/.env` (API keys).
- **Datos de la empresa**: `datos/` completo y la base vectorial (`mi_base_vectorial/`) — se regeneran con la ingesta.
- **Historial de chats**: `web/*.db`.
- **Entornos virtuales**: `Agentes/env/` — se recrea con `pip install -r Requirements.txt`.

Para levantar el proyecto en una máquina nueva desde el repo: clonar, crear venv e instalar `Agentes/Requirements.txt` + `web/requirements.txt`, crear `Agentes/.env` (ver variables en el código), colocar los documentos en `datos/fuentes/` y correr la ingesta.

## Estado actual y pendientes

- ⚠️ **Rebuild pendiente**: la base vectorial actual fue creada antes de la migración a métrica coseno y metadata enriquecida. Correr `ingest_markdown.py --rebuild` activa: métrica coseno, re-ranking sobre texto crudo y keywords muestreadas. Mientras tanto el sistema funciona en modo compatibilidad (con avisos `[WARN]` en consola).
- La carpeta de YouTube está **comentada** en `source_folders` — descomentarla cuando se quiera incorporar ese corpus.
- **Migración a Azure**: planes completos en `docs/Fase1_Preparacion_Azure.docx` (empaquetado, tests, Docker, auth), `docs/Fase2_Lift_Azure.docx` (despliegue administrado) y `docs/Fase3_Nativo_Azure.docx` (AI Search, OpenAI, ingesta por eventos).
- Mapa visual de archivos y comunicación: `docs/arquitectura/mapa_archivos_sistema.svg`.

## Notas de mantenimiento

- El venv (`Agentes/env/`) **no se mueve ni renombra** (rutas absolutas internas). Si se rompe, se recrea desde `Requirements.txt`.
- `Agentes/.env` contiene credenciales reales — nunca versionarlo ni compartirlo.
