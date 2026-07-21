"""Autenticacion: hashing de contraseñas, sesiones firmadas e identidad.

Estandares que sigue este modulo:
  - Contraseñas: PBKDF2-HMAC-SHA256 con 600.000 iteraciones (recomendacion
    OWASP 2023) y salt aleatorio por usuario. El formato guardado es
    autodescriptivo ("pbkdf2_sha256$iteraciones$salt$hash"), asi se puede
    subir el numero de iteraciones en el futuro sin romper hashes viejos.
  - Sesiones: token firmado con HMAC-SHA256 (no cifrado, pero a prueba de
    manipulacion) que viaja en una cookie HttpOnly + SameSite=Lax. El
    servidor no guarda sesiones en memoria: el token se autovalida.
  - Solo libreria estandar (hashlib, hmac, secrets): cero dependencias
    nuevas que instalar o auditar.

============================ COSTURA PARA AZURE ============================
get_current_user() es la UNICA puerta de identidad de toda la app. Los
endpoints (api.py) dependen solo de esta funcion, nunca de cookies ni
tokens directamente.

AZURE (Fase 2): al desplegar detras de Microsoft Entra ID (Easy Auth de
Container Apps / App Service), la plataforma valida el login corporativo
y pasa la identidad ya verificada en la cabecera 'X-MS-CLIENT-PRINCIPAL'.
Para migrar SOLO se reescribe el cuerpo de get_current_user() para leer
esa cabecera (ver _usuario_desde_entra_id, dejada como referencia) — ni
un solo endpoint cambia. El login local de este archivo pasa a ser el
modo de desarrollo, activable con IAPY_AUTH_MODE=local.
===========================================================================
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Request, HTTPException

import chat_store

# --- Parametros de seguridad ---
PBKDF2_ITERACIONES = 600_000            # OWASP 2023 para PBKDF2-SHA256
DURACION_SESION_SEG = 8 * 60 * 60       # 8 horas
NOMBRE_COOKIE = "iapy_sesion"


def _cargar_secret_key() -> bytes:
    """Clave para firmar las cookies de sesion.

    Orden: variable de entorno IAPY_SECRET_KEY (lo que se usara en Azure,
    inyectada desde Key Vault) -> archivo .secret_key junto a la base
    (se genera solo la primera vez, esta en .gitignore).

    AZURE: en produccion SIEMPRE debe venir de IAPY_SECRET_KEY. El archivo
    local es solo para que las sesiones sobrevivan reinicios en desarrollo.
    """
    desde_env = os.getenv("IAPY_SECRET_KEY")
    if desde_env:
        return desde_env.encode("utf-8")

    ruta = Path(__file__).resolve().parent / ".secret_key"
    if ruta.exists():
        return ruta.read_bytes()

    generada = secrets.token_bytes(32)
    ruta.write_bytes(generada)
    print("[auth] Generada una IAPY_SECRET_KEY local en web/.secret_key "
          "(en Azure debe venir de Key Vault via variable de entorno).")
    return generada


_SECRET_KEY = _cargar_secret_key()


# ==========================================================================
# CONTRASEÑAS
# ==========================================================================

def hash_password(password: str) -> str:
    """Deriva un hash seguro de la contraseña (para guardar en la base)."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERACIONES)
    return f"pbkdf2_sha256${PBKDF2_ITERACIONES}${salt.hex()}${dk.hex()}"


def verificar_password(password: str, hash_guardado: str) -> bool:
    """Compara una contraseña contra el hash guardado (tiempo constante)."""
    try:
        algoritmo, iteraciones, salt_hex, hash_hex = hash_guardado.split("$")
        if algoritmo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iteraciones)
        )
        # compare_digest evita ataques de temporizacion (timing attacks).
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ==========================================================================
# TOKENS DE SESION  (firmados, sin estado en el servidor)
# ==========================================================================

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _de_b64(texto: str) -> bytes:
    return base64.urlsafe_b64decode(texto + "=" * (-len(texto) % 4))


def crear_token_sesion(user_id: str) -> str:
    """Genera un token firmado con el id de usuario y una expiracion."""
    payload = {"uid": user_id, "exp": int(time.time()) + DURACION_SESION_SEG}
    cuerpo = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    firma = _b64(hmac.new(_SECRET_KEY, cuerpo.encode("ascii"), hashlib.sha256).digest())
    return f"{cuerpo}.{firma}"


def validar_token_sesion(token: str) -> Optional[str]:
    """Devuelve el user_id si el token es valido y no expiro; si no, None."""
    try:
        cuerpo, firma = token.split(".")
        esperada = _b64(hmac.new(_SECRET_KEY, cuerpo.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(firma, esperada):
            return None  # firma manipulada
        payload = json.loads(_de_b64(cuerpo))
        if payload.get("exp", 0) < time.time():
            return None  # expirado
        return payload.get("uid")
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# ==========================================================================
# IDENTIDAD  (la costura para Entra ID)
# ==========================================================================

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Devuelve el usuario autenticado o lanza 401.

    ESTA es la unica funcion que los endpoints deben usar para saber quien
    esta pidiendo. Hoy lee la cookie de sesion local; en Azure se cambia
    por la cabecera de Entra ID (ver _usuario_desde_entra_id abajo).
    """
    token = request.cookies.get(NOMBRE_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")

    user_id = validar_token_sesion(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Sesion invalida o expirada")

    usuario = chat_store.obtener_usuario_por_id(user_id)
    if not usuario:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    return usuario


async def require_admin(request: Request) -> Dict[str, Any]:
    """Como get_current_user, pero ademas exige rol admin (para endpoints admin)."""
    usuario = await get_current_user(request)
    if usuario.get("rol") != "admin":
        raise HTTPException(status_code=403, detail="Requiere permisos de administrador")
    return usuario


# --- Referencia para la Fase 2 (NO se usa todavia) -----------------------
# Cuando la app corra detras de Entra ID / Easy Auth, get_current_user()
# pasa a tener este cuerpo. Los endpoints no cambian.
#
# def _usuario_desde_entra_id(request: Request) -> Dict[str, Any]:
#     principal_b64 = request.headers.get("X-MS-CLIENT-PRINCIPAL")
#     if not principal_b64:
#         raise HTTPException(status_code=401, detail="No autenticado")
#     datos = json.loads(base64.b64decode(principal_b64))
#     claims = {c["typ"]: c["val"] for c in datos["claims"]}
#     email = claims.get("preferred_username") or claims.get("emails")
#     # Alta automatica la primera vez que un usuario corporativo entra:
#     usuario = chat_store.obtener_usuario_por_email(email)
#     if not usuario:
#         usuario = chat_store.crear_usuario(email, claims.get("name", email), "", rol="usuario")
#     return usuario
