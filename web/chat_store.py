"""Guardado de chats en SQLite.

Muy basico a proposito: dos tablas (chats y mensajes), sin ORM, todo el
SQL a la vista. La base (conversaciones.db) se crea sola al primer uso.

Este modulo es la UNICA puerta a la base — nadie mas ejecuta SQL. Esa
disciplina es lo que hace barata la migracion:

AZURE (Fase 2): sqlite3 -> PostgreSQL Flexible Server. Cambios acotados
a este archivo: psycopg en vez de sqlite3, placeholders %s en vez de ?,
y la cadena de conexion por variable de entorno (con sslmode=require).
El resto del SQL es estandar y migra tal cual.
AZURE (Fase 1, antes de eso): agregar tabla usuarios y columna user_id
en chats para el multiusuario.
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
            CREATE TABLE IF NOT EXISTS usuarios (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                nombre TEXT NOT NULL,
                hash_password TEXT NOT NULL,
                rol TEXT NOT NULL DEFAULT 'usuario',
                creado_en TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                titulo TEXT NOT NULL,
                creado_en TEXT NOT NULL
            )
            """
        )
        # Migracion suave: las bases creadas antes del multiusuario no
        # tienen user_id en chats. Se agrega la columna sin tocar datos;
        # los chats viejos quedan con NULL hasta que un admin los adopte
        # (ver adoptar_chats_sin_usuario, la usa crear_usuario.py).
        columnas = [c[1] for c in con.execute("PRAGMA table_info(chats)")]
        if "user_id" not in columnas:
            con.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
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


# ==========================================================================
# USUARIOS
# ==========================================================================
# Nota: este modulo guarda y consulta usuarios, pero NO sabe nada de
# contraseñas en claro ni de hashing — recibe el hash ya calculado desde
# auth.py. Esa separacion es a proposito: la logica de seguridad vive en
# un solo lugar (auth.py) y la base solo persiste.

def crear_usuario(email: str, nombre: str, hash_password: str, rol: str = "usuario") -> Dict[str, Any]:
    """Inserta un usuario nuevo. Lanza ValueError si el email ya existe."""
    user_id = str(uuid.uuid4())
    creado_en = datetime.now().isoformat(timespec="seconds")
    con = _conectar()
    try:
        try:
            con.execute(
                "INSERT INTO usuarios (id, email, nombre, hash_password, rol, creado_en) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, email.lower().strip(), nombre, hash_password, rol, creado_en),
            )
            con.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Ya existe un usuario con el email {email}")
    finally:
        con.close()
    return {"id": user_id, "email": email.lower().strip(), "nombre": nombre, "rol": rol, "creado_en": creado_en}


def obtener_usuario_por_email(email: str) -> Optional[Dict[str, Any]]:
    """Devuelve el usuario (INCLUYE hash_password) o None. Lo usa el login."""
    con = _conectar()
    try:
        fila = con.execute(
            "SELECT id, email, nombre, hash_password, rol, creado_en FROM usuarios WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()
        return dict(fila) if fila else None
    finally:
        con.close()


def obtener_usuario_por_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Devuelve el usuario SIN el hash (seguro para exponer a la sesion/API)."""
    con = _conectar()
    try:
        fila = con.execute(
            "SELECT id, email, nombre, rol, creado_en FROM usuarios WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(fila) if fila else None
    finally:
        con.close()


def listar_usuarios() -> List[Dict[str, Any]]:
    """Todos los usuarios, sin hashes. Solo para el panel de admin."""
    con = _conectar()
    try:
        filas = con.execute(
            "SELECT id, email, nombre, rol, creado_en FROM usuarios ORDER BY creado_en ASC"
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        con.close()


def cambiar_rol(user_id: str, rol: str) -> None:
    con = _conectar()
    try:
        con.execute("UPDATE usuarios SET rol = ? WHERE id = ?", (rol, user_id))
        con.commit()
    finally:
        con.close()


def contar_usuarios() -> int:
    con = _conectar()
    try:
        return con.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0]
    finally:
        con.close()


def eliminar_usuario(user_id: str) -> None:
    """Borra un usuario y TODOS sus chats/mensajes en cascada."""
    con = _conectar()
    try:
        filas = con.execute("SELECT id FROM chats WHERE user_id = ?", (user_id,)).fetchall()
        for f in filas:
            con.execute("DELETE FROM mensajes WHERE chat_id = ?", (f["id"],))
        con.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
        con.execute("DELETE FROM usuarios WHERE id = ?", (user_id,))
        con.commit()
    finally:
        con.close()


def adoptar_chats_sin_usuario(user_id: str) -> int:
    """Asigna al usuario dado todos los chats huerfanos (user_id NULL).

    Sirve al migrar una base pre-multiusuario: los chats que existian antes
    del login se le adjudican al primer admin. Devuelve cuantos adopto.
    """
    con = _conectar()
    try:
        cur = con.execute(
            "UPDATE chats SET user_id = ? WHERE user_id IS NULL", (user_id,)
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


# ==========================================================================
# CHATS  (siempre acotados a su dueño)
# ==========================================================================

def crear_chat(user_id: str, titulo: str = "Nuevo chat") -> Dict[str, Any]:
    """Crea un chat vacio para `user_id` y devuelve su fila (la GUI la pinta)."""
    chat_id = str(uuid.uuid4())
    creado_en = datetime.now().isoformat(timespec="seconds")
    con = _conectar()
    try:
        con.execute(
            "INSERT INTO chats (id, titulo, creado_en, user_id) VALUES (?, ?, ?, ?)",
            (chat_id, titulo, creado_en, user_id),
        )
        con.commit()
    finally:
        con.close()
    return {"id": chat_id, "titulo": titulo, "creado_en": creado_en}


def listar_chats(user_id: str) -> List[Dict[str, Any]]:
    """Los chats de UN usuario, mas reciente primero."""
    con = _conectar()
    try:
        filas = con.execute(
            "SELECT id, titulo, creado_en FROM chats WHERE user_id = ? ORDER BY creado_en DESC",
            (user_id,),
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        con.close()


def obtener_chat(chat_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Devuelve el chat, o None si no existe.

    Si se pasa `user_id`, el chat solo se devuelve cuando pertenece a ese
    usuario: asi el chequeo de propiedad es automatico en cada endpoint
    (si un usuario pide el chat de otro, obtener_chat devuelve None -> 404).
    Con user_id=None (contexto admin) no se filtra por dueño.
    """
    con = _conectar()
    try:
        if user_id is None:
            fila = con.execute(
                "SELECT id, titulo, creado_en FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
        else:
            fila = con.execute(
                "SELECT id, titulo, creado_en FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
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
