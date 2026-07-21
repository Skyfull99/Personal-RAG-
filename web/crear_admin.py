"""Script de arranque: crea (o resetea) un usuario administrador.

Se corre UNA vez al montar el sistema, o cuando haga falta resetear la
contraseña del admin. Es idempotente: si el usuario ya existe, actualiza
su contraseña y lo deja como admin.

Uso:
    # Interactivo (la contraseña no se ve al escribirla):
    python crear_admin.py correo@ejemplo.com "Nombre Completo"

    # No interactivo (util para automatizar; ojo con el historial de shell):
    python crear_admin.py correo@ejemplo.com "Nombre" --password "LaClave123"

Al crear el primer admin, tambien adopta los chats que existieran antes
del multiusuario (los deja bajo esa cuenta).

AZURE (Fase 2): con Entra ID el alta de usuarios es automatica al primer
login corporativo (ver auth._usuario_desde_entra_id), asi que este script
deja de necesitarse — queda solo como utilidad de administracion local.
"""

import argparse
import getpass
import sys

import chat_store
import auth


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea o resetea un usuario administrador.")
    parser.add_argument("email", help="Correo del administrador")
    parser.add_argument("nombre", help="Nombre completo (entre comillas si tiene espacios)")
    parser.add_argument("--password", help="Contraseña; si se omite, se pide de forma segura")
    args = parser.parse_args()

    chat_store.iniciar_db()

    password = args.password
    if not password:
        password = getpass.getpass("Contraseña para el admin (min. 8 caracteres): ")
        repetir = getpass.getpass("Repite la contraseña: ")
        if password != repetir:
            print("[ERROR] Las contraseñas no coinciden.")
            sys.exit(1)

    if len(password) < 8:
        print("[ERROR] La contraseña debe tener al menos 8 caracteres.")
        sys.exit(1)

    hash_pw = auth.hash_password(password)
    existente = chat_store.obtener_usuario_por_email(args.email)

    if existente:
        # Reset idempotente: nueva contraseña y aseguramos rol admin.
        con = chat_store._conectar()
        try:
            con.execute(
                "UPDATE usuarios SET hash_password = ?, rol = 'admin', nombre = ? WHERE id = ?",
                (hash_pw, args.nombre, existente["id"]),
            )
            con.commit()
        finally:
            con.close()
        user_id = existente["id"]
        print(f"[OK] Usuario '{args.email}' actualizado y confirmado como admin.")
    else:
        usuario = chat_store.crear_usuario(args.email, args.nombre, hash_pw, rol="admin")
        user_id = usuario["id"]
        print(f"[OK] Administrador creado: {args.email}")

    adoptados = chat_store.adoptar_chats_sin_usuario(user_id)
    if adoptados:
        print(f"[OK] {adoptados} chat(s) que existian antes del login quedaron bajo esta cuenta.")

    print("\nYa puedes iniciar sesion en la web con ese correo y contraseña.")


if __name__ == "__main__":
    main()
