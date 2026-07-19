"""Guardado de chats en SQLite.

Muy basico a proposito: dos tablas (chats y mensajes), sin ORM.
La base vive en web/chats.db y se crea sola la primera vez que se usa.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent / "conversaciones.db"


def _conectar() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    # journal_mode=MEMORY evita crear un archivo -journal aparte en disco.
    # Se usa a proposito (en vez del modo por defecto o WAL) porque carpetas
    # sincronizadas como OneDrive/Dropbox a veces bloquean el archivo de
    # journal y provocan errores raros de "disk I/O error". Para un historial
    # de chat personal (no critico) esta es una eleccion segura y simple.
    try:
        con.execute("PRAGMA journal_mode=MEMORY;")
    except sqlite3.OperationalError:
        pass  # se queda con el modo de journal por defecto si ni esto funciona
    return con


def iniciar_db() -> None:
    con = _conectar()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                titulo TEXT NOT NULL,
                creado_en TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS mensajes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                rol TEXT NOT NULL,
                contenido TEXT NOT NULL,
                creado_en TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats (id)
            )
            """
        )
        con.commit()
    finally:
        con.close()


def crear_chat(titulo: str = "Nuevo chat") -> Dict[str, Any]:
    chat_id = str(uuid.uuid4())
    creado_en = datetime.now().isoformat(timespec="seconds")
    con = _conectar()
    try:
        con.execute(
            "INSERT INTO chats (id, titulo, creado_en) VALUES (?, ?, ?)",
            (chat_id, titulo, creado_en),
        )
        con.commit()
    finally:
        con.close()
    return {"id": chat_id, "titulo": titulo, "creado_en": creado_en}


def listar_chats() -> List[Dict[str, Any]]:
    con = _conectar()
    try:
        filas = con.execute(
            "SELECT id, titulo, creado_en FROM chats ORDER BY creado_en DESC"
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        con.close()


def obtener_chat(chat_id: str) -> Optional[Dict[str, Any]]:
    con = _conectar()
    try:
        fila = con.execute(
            "SELECT id, titulo, creado_en FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(fila) if fila else None
    finally:
        con.close()


def renombrar_chat(chat_id: str, titulo: str) -> None:
    con = _conectar()
    try:
        con.execute("UPDATE chats SET titulo = ? WHERE id = ?", (titulo, chat_id))
        con.commit()
    finally:
        con.close()


def eliminar_chat(chat_id: str) -> None:
    con = _conectar()
    try:
        con.execute("DELETE FROM mensajes WHERE chat_id = ?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        con.commit()
    finally:
        con.close()


def agregar_mensaje(chat_id: str, rol: str, contenido: str) -> None:
    creado_en = datetime.now().isoformat(timespec="seconds")
    con = _conectar()
    try:
        con.execute(
            "INSERT INTO mensajes (chat_id, rol, contenido, creado_en) VALUES (?, ?, ?, ?)",
            (chat_id, rol, contenido, creado_en),
        )
        con.commit()
    finally:
        con.close()


def eliminar_ultimo_turno(chat_id: str) -> bool:
    """Elimina el ultimo mensaje del usuario y todo lo que vino despues.

    Es la operacion detras de "editar y reenviar el ultimo prompt": se
    borra el ultimo turno completo (pregunta + su respuesta, si la hay) y
    el frontend reenvia la version editada como un mensaje nuevo.
    Devuelve False si el chat no tiene ningun mensaje de usuario.
    """
    con = _conectar()
    try:
        fila = con.execute(
            "SELECT id FROM mensajes WHERE chat_id = ? AND rol = 'user' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if not fila:
            return False
        con.execute(
            "DELETE FROM mensajes WHERE chat_id = ? AND id >= ?",
            (chat_id, fila["id"]),
        )
        con.commit()
        return True
    finally:
        con.close()


def obtener_mensajes(chat_id: str) -> List[Dict[str, Any]]:
    con = _conectar()
    try:
        filas = con.execute(
            "SELECT rol, contenido, creado_en FROM mensajes WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        con.close()
