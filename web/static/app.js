// GUI basica de chat para el RAG local.
// Maneja: tema claro/oscuro (guardado en localStorage), lista de chats
// guardados (en el servidor, via SQLite), y el envio/recepcion de mensajes.

document.addEventListener("DOMContentLoaded", () => {

    const btnTema = document.getElementById("btn-tema");
    const btnNuevoChat = document.getElementById("btn-nuevo-chat");
    const listaChats = document.getElementById("lista-chats");
    const divMensajes = document.getElementById("mensajes");
    const formMensaje = document.getElementById("form-mensaje");
    const entrada = document.getElementById("entrada");
    const btnEnviar = document.getElementById("btn-enviar");

    let chatActualId = null;

    // ---------- Tema claro/oscuro ----------

    function aplicarTemaGuardado() {
        // El modo noche es el predeterminado de la marca; "claro" solo si
        // el usuario lo eligio explicitamente con el boton de tema.
        const temaGuardado = localStorage.getItem("rag_tema") || "oscuro";
        document.body.classList.toggle("tema-oscuro", temaGuardado === "oscuro");
    }

    function alternarTema() {
        const esOscuroAhora = document.body.classList.toggle("tema-oscuro");
        localStorage.setItem("rag_tema", esOscuroAhora ? "oscuro" : "claro");
    }

    btnTema.addEventListener("click", alternarTema);
    aplicarTemaGuardado();

    // ---------- Lista de chats guardados ----------

    async function cargarListaChats() {
        const resp = await fetch("/api/chats");
        const chats = await resp.json();

        listaChats.innerHTML = "";

        if (chats.length === 0) {
            const vacio = document.createElement("p");
            vacio.className = "placeholder";
            vacio.style.fontSize = "0.85rem";
            vacio.textContent = "Sin chats todavia.";
            listaChats.appendChild(vacio);
            return;
        }

        for (const chat of chats) {
            const item = document.createElement("div");
            item.className = "item-chat" + (chat.id === chatActualId ? " activo" : "");

            const nombre = document.createElement("span");
            nombre.className = "nombre-chat";
            nombre.textContent = chat.titulo;
            nombre.addEventListener("click", () => seleccionarChat(chat.id));

            const btnRenombrar = document.createElement("button");
            btnRenombrar.className = "btn-renombrar";
            btnRenombrar.textContent = "✎";
            btnRenombrar.title = "Renombrar chat";
            btnRenombrar.addEventListener("click", (e) => {
                e.stopPropagation();
                activarRenombrado(nombre, chat);
            });

            const btnBorrar = document.createElement("button");
            btnBorrar.className = "btn-borrar";
            btnBorrar.textContent = "×";
            btnBorrar.title = "Borrar chat";
            btnBorrar.addEventListener("click", (e) => {
                e.stopPropagation();
                borrarChat(chat.id);
            });

            item.appendChild(nombre);
            item.appendChild(btnRenombrar);
            item.appendChild(btnBorrar);
            listaChats.appendChild(item);
        }
    }

    // ---------- Renombrar un chat (edicion inline en la barra lateral) ----------

    function activarRenombrado(nombreSpan, chat) {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "input-renombrar";
        input.value = chat.titulo;
        input.maxLength = 100;
        nombreSpan.replaceWith(input);
        input.focus();
        input.select();

        let cancelado = false;

        const terminar = async () => {
            const nuevo = input.value.trim();
            if (!cancelado && nuevo && nuevo !== chat.titulo) {
                await fetch(`/api/chats/${chat.id}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ titulo: nuevo }),
                });
            }
            // Repintar la lista siempre (restaura el span si se cancelo).
            await cargarListaChats();
        };

        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") input.blur();
            if (e.key === "Escape") {
                cancelado = true;
                input.blur();
            }
            e.stopPropagation();
        });
        input.addEventListener("click", (e) => e.stopPropagation());
        input.addEventListener("blur", terminar);
    }

    async function crearNuevoChat() {
        const resp = await fetch("/api/chats", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ titulo: "Nuevo chat" }),
        });
        const chat = await resp.json();
        chatActualId = chat.id;
        divMensajes.innerHTML = "";
        mostrarPlaceholder("Escribe tu primera pregunta para este chat.");
        await cargarListaChats();
        entrada.focus();
    }

    async function borrarChat(chatId) {
        await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
        if (chatId === chatActualId) {
            chatActualId = null;
            divMensajes.innerHTML = "";
            mostrarPlaceholder("Selecciona un chat de la izquierda o crea uno nuevo para empezar.");
        }
        await cargarListaChats();
    }

    async function seleccionarChat(chatId) {
        chatActualId = chatId;
        await cargarListaChats();

        const resp = await fetch(`/api/chats/${chatId}/mensajes`);
        const mensajes = await resp.json();

        divMensajes.innerHTML = "";
        if (mensajes.length === 0) {
            mostrarPlaceholder("Escribe tu primera pregunta para este chat.");
        } else {
            for (const m of mensajes) {
                agregarBurbuja(m.rol, m.contenido);
            }
            desplazarAlFinal();
            actualizarBotonEditar();
        }
    }

    btnNuevoChat.addEventListener("click", crearNuevoChat);

    // ---------- Mensajes ----------

    function mostrarPlaceholder(texto) {
        divMensajes.innerHTML = `<p class="placeholder">${texto}</p>`;
    }

    function limpiarPlaceholder() {
        const placeholder = divMensajes.querySelector(".placeholder");
        if (placeholder) placeholder.remove();
    }

    // ---------- Render de Markdown (sin librerias externas) ----------
    // El RAG responde en Markdown (##, **negrita**, listas). Esto lo
    // convierte a HTML basico para que se vea limpio en vez de mostrar
    // los simbolos "##" y "**" tal cual. Escapa el texto original primero
    // para que el contenido de los documentos no pueda inyectar HTML.

    function escaparHTML(texto) {
        const div = document.createElement("div");
        div.textContent = texto;
        return div.innerHTML;
    }

    function renderizarMarkdown(texto) {
        let html = escaparHTML(texto);

        // Encabezados (### antes que ## antes que #, para no pisarlos).
        // Se fuerza una linea en blanco antes/despues de cada uno: si el
        // encabezado viene pegado a su parrafo (un solo salto de linea, no
        // dos), el separador de parrafos de mas abajo lo tratada como un
        // solo bloque y no envolvia el texto siguiente en <p>.
        html = html.replace(/^### (.*)$/gm, "\n\n<h3>$1</h3>\n\n");
        html = html.replace(/^## (.*)$/gm, "\n\n<h2>$1</h2>\n\n");
        html = html.replace(/^# (.*)$/gm, "\n\n<h1>$1</h1>\n\n");

        // Separador horizontal (*** o ---), mismo motivo que arriba.
        html = html.replace(/^(\*\*\*|---)$/gm, "\n\n<hr>\n\n");

        // Negrita
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

        // Listas: agrupa lineas consecutivas que empiecen con "* " o "- ".
        // Igual que arriba, se aisla con lineas en blanco para que no quede
        // un <ul> metido dentro de un <p> (HTML invalido).
        html = html.replace(/(?:^|\n)((?:[*-] .*(?:\n|$))+)/g, (bloque) => {
            const items = bloque
                .trim()
                .split("\n")
                .map((linea) => `<li>${linea.replace(/^[*-]\s+/, "")}</li>`)
                .join("");
            return `\n\n<ul>${items}</ul>\n\n`;
        });

        // Parrafos: separa por doble salto de linea; no envuelve en <p>
        // los bloques que ya son encabezado/lista/separador.
        html = html
            .split(/\n{2,}/)
            .map((bloque) => {
                const t = bloque.trim();
                if (!t) return "";
                if (/^<(h1|h2|h3|ul|hr)/.test(t)) return t;
                return `<p>${t.replace(/\n/g, "<br>")}</p>`;
            })
            .join("");

        return html;
    }

    function agregarBurbuja(rol, texto, cargando = false) {
        limpiarPlaceholder();
        const burbuja = document.createElement("div");
        burbuja.className = "burbuja " + (rol === "user" ? "usuario" : "asistente") + (cargando ? " cargando" : "");
        if (rol === "user" || cargando) {
            burbuja.textContent = texto;
        } else {
            burbuja.innerHTML = renderizarMarkdown(texto);
        }
        if (rol === "user") {
            // Guarda el texto original para la funcion "editar y reenviar".
            burbuja.dataset.texto = texto;
        }
        divMensajes.appendChild(burbuja);
        desplazarAlFinal();
        return burbuja;
    }

    function desplazarAlFinal() {
        divMensajes.scrollTop = divMensajes.scrollHeight;
    }

    // Burbuja de espera con animacion en dos fases (solo CSS, ver style.css):
    // 1) una lupa barre de lado a lado mientras "busca";
    // 2) a los ~5s se transforma en un ojo que mira a los lados y pestañea
    //    mientras "analiza". El primer fragmento real del stream la reemplaza.
    function crearBurbujaCargando() {
        limpiarPlaceholder();
        const burbuja = document.createElement("div");
        burbuja.className = "burbuja asistente cargando";
        burbuja.innerHTML = `
            <div class="indicador-carga">
                <span class="icono-carga">
                    <span class="lupa"></span>
                    <span class="ojo"><span class="pupila"></span></span>
                </span>
                <span class="textos-carga">
                    <span class="texto-analizando">Analizando los documentos encontrados...</span>
                    <span class="texto-buscando">Buscando en la base de conocimiento...</span>
                </span>
            </div>`;
        divMensajes.appendChild(burbuja);
        desplazarAlFinal();
        return burbuja;
    }

    function bloquearEntrada(bloquear) {
        btnEnviar.disabled = bloquear;
        entrada.disabled = bloquear;
        if (!bloquear) entrada.focus();
    }

    // Nucleo compartido entre "enviar mensaje nuevo" y "editar y reenviar":
    // pinta las burbujas y consume el stream SSE del servidor.
    async function ejecutarPregunta(texto) {
        bloquearEntrada(true);
        quitarBotonesEditar(); // no se edita mientras se esta generando

        agregarBurbuja("user", texto);
        const burbujaCargando = crearBurbujaCargando();

        try {
            // Endpoint streaming (SSE): la respuesta llega por fragmentos
            // y se pinta en vivo, en vez de esperarla completa mirando un
            // spinner durante decenas de segundos.
            const resp = await fetch(`/api/chats/${chatActualId}/mensajes/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texto: texto }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Error del servidor (${resp.status})`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let textoRespuesta = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                // Los eventos SSE vienen separados por doble salto de linea.
                // Lo que quede incompleto en el buffer espera al proximo chunk.
                const partes = buffer.split("\n\n");
                buffer = partes.pop();

                for (const parte of partes) {
                    const linea = parte.trim();
                    if (!linea.startsWith("data: ")) continue;
                    const evento = JSON.parse(linea.slice(6));

                    if (evento.error) {
                        throw new Error(evento.error);
                    }
                    if (evento.delta !== undefined) {
                        if (textoRespuesta === "") {
                            // Primer fragmento: deja de ser "cargando".
                            burbujaCargando.classList.remove("cargando");
                        }
                        textoRespuesta += evento.delta;
                        burbujaCargando.innerHTML = renderizarMarkdown(textoRespuesta);
                        desplazarAlFinal();
                    }
                    // evento.fin: nada que hacer, el reader terminara solo.
                }
            }

            // El titulo del chat pudo haber cambiado (se autogenera con el
            // primer mensaje), asi que refrescamos la lista de la izquierda.
            await cargarListaChats();

        } catch (error) {
            burbujaCargando.textContent = "Error: " + error.message;
            burbujaCargando.classList.remove("cargando");
        } finally {
            bloquearEntrada(false);
            actualizarBotonEditar();
        }
    }

    async function enviarMensaje(evento) {
        evento.preventDefault();
        const texto = entrada.value.trim();
        if (!texto) return;

        // Si todavia no hay un chat activo, se crea uno automaticamente.
        if (!chatActualId) {
            await crearNuevoChat();
        }

        entrada.value = "";
        entrada.style.height = "auto";
        await ejecutarPregunta(texto);
    }

    // ---------- Editar y reenviar el ultimo prompt ----------
    // Como en las UIs modernas de LLM: solo el ULTIMO mensaje del usuario
    // es editable. Al reenviarlo, el turno viejo (pregunta + respuesta) se
    // borra del historial y el texto editado se procesa como mensaje nuevo.

    function quitarBotonesEditar() {
        divMensajes.querySelectorAll(".btn-editar").forEach((b) => b.remove());
    }

    function actualizarBotonEditar() {
        quitarBotonesEditar();
        const burbujasUsuario = divMensajes.querySelectorAll(".burbuja.usuario");
        if (burbujasUsuario.length === 0) return;
        const ultima = burbujasUsuario[burbujasUsuario.length - 1];
        if (ultima.classList.contains("editando")) return;

        const btn = document.createElement("button");
        btn.className = "btn-editar";
        btn.textContent = "✎";
        btn.title = "Editar y reenviar";
        btn.addEventListener("click", () => activarEdicion(ultima));
        ultima.appendChild(btn);
    }

    function activarEdicion(burbuja) {
        const original = burbuja.dataset.texto || "";
        quitarBotonesEditar();
        burbuja.classList.add("editando");
        burbuja.innerHTML = "";

        const areaEdicion = document.createElement("textarea");
        areaEdicion.className = "textarea-edicion";
        areaEdicion.value = original;
        areaEdicion.rows = 1;

        // El textarea crece con el contenido (sin barras de scroll ni
        // flechas), para que se sienta como editar el texto de la burbuja.
        const ajustarAltura = () => {
            areaEdicion.style.height = "auto";
            areaEdicion.style.height = areaEdicion.scrollHeight + "px";
        };
        areaEdicion.addEventListener("input", ajustarAltura);

        const acciones = document.createElement("div");
        acciones.className = "acciones-edicion";

        const btnCancelar = document.createElement("button");
        btnCancelar.type = "button";
        btnCancelar.className = "btn-cancelar-edicion";
        btnCancelar.textContent = "Cancelar";

        const btnReenviar = document.createElement("button");
        btnReenviar.type = "button";
        btnReenviar.className = "btn-guardar-edicion";
        btnReenviar.textContent = "Enviar";

        const cancelar = () => {
            burbuja.classList.remove("editando");
            burbuja.innerHTML = "";
            burbuja.textContent = original;
            burbuja.dataset.texto = original;
            actualizarBotonEditar();
        };

        btnCancelar.addEventListener("click", cancelar);
        btnReenviar.addEventListener("click", () => {
            const nuevo = areaEdicion.value.trim();
            if (!nuevo) return;
            reenviarEditado(nuevo, burbuja);
        });
        areaEdicion.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.key === "Enter") btnReenviar.click();
            if (e.key === "Escape") cancelar();
        });

        acciones.appendChild(btnCancelar);
        acciones.appendChild(btnReenviar);
        burbuja.appendChild(areaEdicion);
        burbuja.appendChild(acciones);
        ajustarAltura(); // necesita estar en el DOM para medir scrollHeight
        areaEdicion.focus();
        areaEdicion.setSelectionRange(areaEdicion.value.length, areaEdicion.value.length);
    }

    async function reenviarEditado(nuevoTexto, burbujaUsuario) {
        bloquearEntrada(true);
        try {
            // 1. Borrar en el servidor el ultimo turno (pregunta + respuesta).
            const resp = await fetch(`/api/chats/${chatActualId}/ultimo-turno`, {
                method: "DELETE",
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Error del servidor (${resp.status})`);
            }
        } catch (error) {
            bloquearEntrada(false);
            alert("No se pudo editar el mensaje: " + error.message);
            return;
        }

        // 2. Quitar del DOM la burbuja editada y todo lo que le sigue
        //    (la respuesta vieja del asistente).
        while (burbujaUsuario.nextElementSibling) {
            burbujaUsuario.nextElementSibling.remove();
        }
        burbujaUsuario.remove();

        // 3. Reenviar el texto editado como un mensaje nuevo (streaming).
        await ejecutarPregunta(nuevoTexto);
    }

    formMensaje.addEventListener("submit", enviarMensaje);

    // Ctrl+Enter envia; Enter solo agrega una linea nueva.
    entrada.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            e.preventDefault();
            formMensaje.requestSubmit();
        }
    });

    // El textarea crece un poco con el contenido, sin volverse gigante.
    entrada.addEventListener("input", () => {
        entrada.style.height = "auto";
        entrada.style.height = Math.min(entrada.scrollHeight, 160) + "px";
    });

    // ---------- Arranque ----------
    cargarListaChats();
});
