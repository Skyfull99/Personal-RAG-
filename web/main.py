r"""Punto de entrada de la aplicacion web.

Para correrlo (usando el venv de Agentes, que ya tiene ollama/chromadb/etc.):
    1. cd Agentes
    2. .\env\Scripts\python.exe -m pip install -r ..\web\requirements.txt
    3. cd ..\web
    4. ..\Agentes\env\Scripts\python.exe -m uvicorn main:app --reload

Luego abre en el navegador: http://localhost:8000

--reload hace que el servidor se reinicie automaticamente cada vez
que guardas un cambio en el codigo. Muy util para desarrollar (pero hace
que rag_service.py se vuelva a inicializar en cada recarga).
"""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import chat_store

# Nos aseguramos de que la base de chats exista ANTES de que cualquier
# endpoint intente usarla.
chat_store.iniciar_db()

# Importamos los endpoints definidos en api.py. Esto tambien dispara la
# inicializacion de rag_service.py (carga del indice BM25, ChromaDB, etc.),
# que puede tardar unos segundos — es intencional que pase aqui, al
# arrancar el servidor, y no en la primera pregunta del usuario.
from api import router as api_router

# Carpeta donde vive este archivo (la carpeta web/)
BASE_DIR = Path(__file__).resolve().parent

# Creamos la aplicacion FastAPI
app = FastAPI(title="RAG Web GUI")

# Conectamos la carpeta /static/ a la URL /static
# Asi cuando el navegador pide /static/style.css FastAPI le devuelve el archivo
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Motor de plantillas: convierte index.html (Jinja2) en una respuesta HTML
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Conectamos todos los endpoints definidos en api.py bajo el prefijo /api
# Es decir: /api/health, /api/chats, /api/chats/{id}/mensajes, etc.
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Sirve la pagina principal (la GUI de chat) cuando el navegador visita http://localhost:8000/"""
    # Nota: en versiones nuevas de Starlette, TemplateResponse ya no acepta
    # el "request" metido dentro del diccionario de contexto (eso rompia con
    # un TypeError raro dentro del cache de Jinja2). Ahora va como primer
    # argumento posicional, separado del contexto.
    return templates.TemplateResponse(request, "index.html", {})
