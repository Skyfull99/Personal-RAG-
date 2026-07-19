import argparse
import hashlib
import json
import os
import time
import chromadb
import ollama
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# Rutas ancladas a la ubicacion de ESTE archivo (no al directorio de
# trabajo): la ingesta funciona igual sin importar desde donde se ejecute,
# y el proyecto no depende de rutas absolutas de una maquina concreta.
_BASE_DIR = Path(__file__).resolve().parent        # .../Agentes
_RAIZ_PROYECTO = _BASE_DIR.parent                  # .../IAPY


@dataclass
class IngestionConfig:
    db_path: str = str(_BASE_DIR / "mi_base_vectorial")
    collection_name: str = "mis_videos_estructurados"
    parent_store_path: str = str(_BASE_DIR / "mi_base_vectorial" / "parent_store.json")
    manifest_path: str = str(_BASE_DIR / "mi_base_vectorial" / "ingest_manifest.json")
    embedding_model: str = "nomic-embed-text"
    llm_enricher_model: str = "gemma4:e4b"
    llm_situator_model: str = "gemma4:e4b"
    # Soporta varias carpetas fuente: los documentos de la empresa y el
    # conocimiento de video (YouTube) se ingestan juntos en la misma base.
    # NOTA: con la ingesta incremental es seguro descomentar la carpeta de
    # YouTube cuando se quiera: solo se procesaran los archivos nuevos o
    # modificados, y tener una carpeta comentada NO borra nada de la base
    # (los archivos ausentes solo se eliminan si se corre con --prune).
    source_folders: List[str] = field(default_factory=lambda: [
        # str(_RAIZ_PROYECTO / "datos" / "fuentes" / "youtube-markdown"),
        str(_RAIZ_PROYECTO / "datos" / "fuentes" / "archivos-alta"),
    ])
    child_chunk_size: int = 450
    child_chunk_overlap: int = 90
    batch_size: int = 50


def _atomic_write_json(path: Path, data: Any) -> None:
    """Escribe JSON de forma atomica: primero a un .tmp y luego reemplaza.

    Asi un corte a mitad de escritura nunca deja el archivo corrupto:
    o queda la version anterior completa, o la nueva completa.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


class IngestManifest:
    """Registro persistente de archivos ya ingestados (ingesta incremental).

    Por cada archivo fuente guarda:
      - sha256:     hash del contenido (la verdad definitiva de "¿cambio?")
      - mtime/size: fecha de modificacion y tamaño (chequeo rapido: si no
                    cambiaron, ni siquiera se calcula el hash)
      - ingested_at: cuando se ingesto por ultima vez
      - parent_ids / child_ids: los IDs exactos que este archivo dejo en el
                    parent store y en ChromaDB, para poder borrarlos de forma
                    quirurgica cuando el archivo cambie o se elimine.
    """
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"[OK] Manifest cargado: {len(self._data)} archivo(s) ya ingestados.")
        else:
            print("[INFO] No hay manifest previo: se creara uno nuevo (primera ingesta incremental).")

    def get(self, filename: str) -> Dict[str, Any] | None:
        return self._data.get(filename)

    def update(self, filename: str, entry: Dict[str, Any]) -> None:
        self._data[filename] = entry

    def remove(self, filename: str) -> None:
        self._data.pop(filename, None)

    def filenames(self) -> List[str]:
        return list(self._data.keys())

    def save(self) -> None:
        _atomic_write_json(self.path, self._data)


class ParentStore:
    """Almacen persistente de chunks padre (seccion bajo cada ##).

    Carga el JSON existente al arrancar (para ingesta incremental) y se
    guarda de forma atomica tras procesar cada archivo. El RAG agent lo
    carga al inicio para resolver children -> parents al responder.
    """
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"[OK] Parent Store existente cargado: {len(self._data)} parents.")

    def add(self, parent_id: str, content: str, metadata: Dict[str, Any]) -> None:
        self._data[parent_id] = {"content": content, "metadata": metadata}

    def remove_many(self, parent_ids: List[str]) -> None:
        for pid in parent_ids:
            self._data.pop(pid, None)

    def save(self) -> None:
        _atomic_write_json(self.path, self._data)


class MetadataEnricher:
    """Responsabilidad: extraer keywords globales del documento (expande acronimos)."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.options = {"temperature": 0.0, "num_ctx": 4096}

    @staticmethod
    def _muestrear(content: str, max_chars: int = 3500) -> str:
        """Muestra representativa del documento: inicio + medio + final.

        Antes se usaban solo los primeros 3500 caracteres, asi que en una
        transcripcion de una hora las keywords reflejaban unicamente los
        primeros minutos. Con tres muestras repartidas, los temas que
        aparecen a mitad o al final del documento tambien quedan cubiertos.
        """
        if len(content) <= max_chars:
            return content
        tercio = max_chars // 3
        medio = len(content) // 2
        return (
            content[:tercio]
            + "\n[...]\n"
            + content[medio - tercio // 2: medio + tercio // 2]
            + "\n[...]\n"
            + content[-tercio:]
        )

    def extract_global_keywords(self, content: str) -> str:
        prompt = f"""Eres un analista de datos tecnicos.
        Analiza el siguiente documento y extrae las 15 palabras clave principales.
        REGLA DE ORO: Si encuentras acronimos tecnicos (ej. RAG, API, MCP, LLM), DEBES expandirlos en tu lista (ej. Escribe "RAG, Retrieval Augmented Generation").

        Devuelve UNICAMENTE una lista separada por comas. Nada de introducciones.

        Documento (muestras del inicio, medio y final, separadas por [...]):
        {self._muestrear(content)}
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self.options,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] Error en LLM Enricher: {e}")
            return "No keywords available"


class ContextualSituator:
    """Responsabilidad: Contextual Retrieval (Anthropic, sept 2024).

    Para cada child genera 2-3 frases que situan el fragmento dentro del
    parent y de los conceptos globales del documento. Se prependen al
    contenido antes de vectorizar para mejorar la recuperacion.

    Optimizacion vs paper original: en vez de pasar el documento entero
    al LLM (caro en modelos locales), pasamos el parent + keywords globales.
    El parent ya da contexto local rico; las keywords cubren el global.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.options = {"temperature": 0.0, "num_ctx": 4096}

    def situate(self, parent_content: str, global_keywords: str, child_content: str) -> str:
        prompt = f"""Eres un asistente que escribe contextos breves para mejorar la busqueda en sistemas RAG.

<seccion_padre>
{parent_content[:3000]}
</seccion_padre>

<conceptos_globales_del_documento>
{global_keywords}
</conceptos_globales_del_documento>

<fragmento>
{child_content}
</fragmento>

Escribe 2-3 frases (maximo 60 palabras) que SITUEN el fragmento dentro de la seccion padre y los conceptos globales del documento. NO expliques ni resumas el contenido del fragmento, SOLO situalo (ej. "Este fragmento describe el paso 2 de X dentro de la seccion sobre Y. Se relaciona con los conceptos Z y W del documento."). Responde sin introducciones.
"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self.options,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] Error en LLM Situator: {e}")
            return ""


class VectorDBManager:
    """Acceso a ChromaDB.

    IMPORTANTE: ya NO borra la coleccion al arrancar (antes lo hacia, y un
    fallo a mitad de ingesta dejaba la base vacia). Ahora la coleccion se
    abre tal cual esta y solo se tocan los chunks del archivo que se esta
    (re)ingestando. El borrado total solo ocurre con --rebuild explicito.
    """
    # Metrica de distancia explicita. Chroma usa L2 por defecto si no se
    # especifica; para embeddings de texto como nomic-embed-text el estandar
    # es similitud coseno, que ademas da scores interpretables (1 = identico,
    # 0 = sin relacion). NOTA: la metrica solo se aplica al CREAR la
    # coleccion — una coleccion vieja en L2 no se puede convertir en sitio,
    # hay que regenerarla con --rebuild.
    COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(self, db_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.COLLECTION_METADATA
        )
        space = (self.collection.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            print(f"[WARN] La coleccion existente usa metrica '{space}' (creada antes del cambio a coseno).")
            print("       Todo sigue funcionando, pero se recomienda migrar corriendo con --rebuild.")

    def recreate_collection(self) -> None:
        """Solo para --rebuild: borra y recrea la coleccion completa."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print("[INFO] Coleccion anterior eliminada (--rebuild).")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.COLLECTION_METADATA
        )

    def upsert_batch(self, embeddings, documents, metadatas, ids):
        # upsert (en vez de add) hace la operacion idempotente: si un id ya
        # existe (p.ej. tras un corte a mitad de un archivo), se sobrescribe
        # en vez de fallar por duplicado.
        self.collection.upsert(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

    def delete_ids(self, ids: List[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)


class ParentChildChunker:
    """Chunking jerarquico: seccion ## = parent, sub-chunks = children."""
    def __init__(self, child_chunk_size: int, child_chunk_overlap: int):
        self.headers_to_split_on = [("##", "Contexto_Semantico")]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split_into_parents(self, content: str):
        return self.header_splitter.split_text(content)

    def split_parent_into_children(self, parent_doc):
        return self.child_splitter.split_documents([parent_doc])


class DirectoryLoader:
    """Descubre archivos .md en una o varias carpetas fuente."""
    def __init__(self, directory_paths: List[str], extension: str = ".md"):
        self.directory_paths = [Path(p) for p in directory_paths]
        self.extension = extension

    def discover_files(self) -> List[Path]:
        archivos: List[Path] = []
        for directory_path in self.directory_paths:
            if not directory_path.exists():
                print(f"[WARN] Carpeta no encontrada, se omite: {directory_path}")
                continue
            encontrados = sorted(directory_path.glob(f"*{self.extension}"))
            print(f"[INFO] {len(encontrados)} archivo(s) {self.extension} en: {directory_path}")
            archivos.extend(encontrados)

        if not archivos:
            raise FileNotFoundError(
                f"No se encontraron archivos {self.extension} en ninguna de estas carpetas: {self.directory_paths}"
            )
        return archivos


class IngestionPipeline:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.db_manager = VectorDBManager(config.db_path, config.collection_name)
        self.embeddings_provider = OllamaEmbeddings(model=config.embedding_model)
        self.chunker = ParentChildChunker(config.child_chunk_size, config.child_chunk_overlap)
        self.loader = DirectoryLoader(config.source_folders)
        self.enricher = MetadataEnricher(config.llm_enricher_model)
        self.situator = ContextualSituator(config.llm_situator_model)
        self.parent_store = ParentStore(Path(config.parent_store_path))
        self.manifest = IngestManifest(Path(config.manifest_path))

    def run(self, prune: bool = False) -> None:
        print("[INFO] Buscando documentos en:")
        for folder in self.config.source_folders:
            print(f"       - {folder}")
        start_time = time.time()

        archivos = self.loader.discover_files()

        sin_cambios = 0
        nuevos = 0
        reingestados = 0

        for file_path in archivos:
            filename = file_path.name
            stat = file_path.stat()
            entry = self.manifest.get(filename)

            # Chequeo rapido: si fecha de modificacion y tamaño no cambiaron
            # desde la ultima ingesta, se salta sin ni siquiera leer el archivo.
            if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
                sin_cambios += 1
                continue

            # El mtime cambio (o es un archivo nuevo): el hash del contenido
            # decide. Asi, un archivo "tocado" pero identico no se reprocesa.
            file_hash = _sha256_file(file_path)
            if entry and entry.get("sha256") == file_hash:
                # Mismo contenido, solo cambio el mtime: actualizar el
                # manifest para que el chequeo rapido funcione la proxima vez.
                entry["mtime"] = stat.st_mtime
                entry["size"] = stat.st_size
                self.manifest.update(filename, entry)
                self.manifest.save()
                sin_cambios += 1
                continue

            if entry:
                print(f"\n[DOC] Modificado, re-ingestando: {filename}")
                reingestados += 1
            else:
                print(f"\n[DOC] Nuevo, ingestando: {filename}")
                nuevos += 1

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._ingest_file(filename, content, file_hash, stat)

        self._report_orphans(archivos, prune)

        print(f"\n[STATS] {nuevos} nuevo(s) | {reingestados} re-ingestado(s) | {sin_cambios} sin cambios (saltados).")
        print(f"[OK] Ingesta completada. Tiempo: {time.time() - start_time:.2f}s")

    def _ingest_file(self, filename: str, content: str, file_hash: str, stat) -> None:
        """Procesa UN archivo completo y deja base + stores consistentes.

        El orden importa para la robustez ante cortes:
          1. Borrar los chunks viejos del archivo (si existia una version previa).
          2. Vectorizar e insertar (upsert) los chunks nuevos.
          3. Guardar parent store y manifest.
        Si el proceso se corta en cualquier punto intermedio, el manifest
        todavia apunta a la version vieja del archivo, asi que la proxima
        corrida detecta el hash distinto y lo re-ingesta desde cero. Los
        demas archivos nunca se ven afectados.
        """
        entry_previa = self.manifest.get(filename)
        if entry_previa:
            self.db_manager.delete_ids(entry_previa.get("child_ids", []))
            self.parent_store.remove_many(entry_previa.get("parent_ids", []))

        clean_source = filename.replace("_Markdown.md", "").replace("_", " ")

        global_keywords = self.enricher.extract_global_keywords(content)
        print(f"   Keywords: {global_keywords[:80]}...")

        parent_docs = self.chunker.split_into_parents(content)
        print(f"   {len(parent_docs)} parents detectados.")

        children_del_archivo: List[Dict[str, Any]] = []
        parent_ids: List[str] = []

        for p_idx, parent in enumerate(parent_docs):
            parent_id = f"{Path(filename).stem}_p{p_idx}"
            topic = parent.metadata.get("Contexto_Semantico", "General")

            self.parent_store.add(parent_id, parent.page_content, {
                "source": clean_source,
                "source_file": filename,
                "topic": topic,
                "global_keywords": global_keywords,
                "parent_index": p_idx,
            })
            parent_ids.append(parent_id)

            children = self.chunker.split_parent_into_children(parent)

            for c_idx, child in enumerate(children):
                situational_context = self.situator.situate(
                    parent.page_content, global_keywords, child.page_content
                )

                enriched_text = (
                    f"FUENTE: {clean_source}\n"
                    f"TEMA: {topic}\n"
                    f"CONCEPTOS CLAVE: {global_keywords}\n"
                    f"CONTEXTO SITUACIONAL: {situational_context}\n\n"
                    f"CONTENIDO:\n{child.page_content}"
                )

                children_del_archivo.append({
                    "content": enriched_text,
                    "metadata": {
                        "source": clean_source,
                        "source_file": filename,
                        "topic": topic,
                        "global_keywords": global_keywords,
                        "parent_id": parent_id,
                        "child_index": c_idx,
                        "situational_context": situational_context[:500],
                        # Contenido crudo del child (sin el header FUENTE/TEMA/
                        # keywords). El texto enriquecido es el que se vectoriza,
                        # pero el cross-encoder de re-ranking puntua sobre este
                        # crudo: el header, repetido en todos los chunks del
                        # mismo documento, sesga sus scores.
                        "child_content": child.page_content,
                    },
                    "id": f"{parent_id}_c{c_idx}",
                })

            print(f"      Parent {p_idx + 1}/{len(parent_docs)} ({topic[:40]}): {len(children)} children con contexto.")

        print(f"   Vectorizando {len(children_del_archivo)} children...")
        self._process_in_batches(children_del_archivo)

        # Recien ahora, con los chunks ya insertados, se persisten los
        # stores. Guardar por archivo (y no al final de todo) hace que un
        # corte solo pierda el archivo en curso, nunca lo ya procesado.
        self.parent_store.save()
        self.manifest.update(filename, {
            "sha256": file_hash,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            "parent_ids": parent_ids,
            "child_ids": [c["id"] for c in children_del_archivo],
        })
        self.manifest.save()
        print(f"   [OK] {filename}: {len(parent_ids)} parents / {len(children_del_archivo)} children guardados.")

    def _report_orphans(self, archivos_presentes: List[Path], prune: bool) -> None:
        """Detecta archivos que estan en el manifest pero ya no en las carpetas.

        Por defecto SOLO avisa (no borra nada): una carpeta fuente comentada
        en la config haria "desaparecer" todos sus archivos, y borrarlos
        automaticamente destruiria conocimiento valido. Con --prune se
        eliminan de verdad (chunks, parents y entrada del manifest).
        """
        presentes = {p.name for p in archivos_presentes}
        huerfanos = [f for f in self.manifest.filenames() if f not in presentes]
        if not huerfanos:
            return

        if not prune:
            print(f"\n[WARN] {len(huerfanos)} archivo(s) del manifest ya no estan en las carpetas fuente:")
            for f in huerfanos:
                print(f"       - {f}")
            print("       Sus chunks SIGUEN en la base. Para eliminarlos, corre con --prune.")
            return

        print(f"\n[INFO] --prune: eliminando {len(huerfanos)} archivo(s) ausentes de la base...")
        for f in huerfanos:
            entry = self.manifest.get(f)
            self.db_manager.delete_ids(entry.get("child_ids", []))
            self.parent_store.remove_many(entry.get("parent_ids", []))
            self.manifest.remove(f)
            print(f"       - Eliminado: {f}")
        self.parent_store.save()
        self.manifest.save()

    def rebuild(self) -> None:
        """Regeneracion total desde cero (el viejo comportamiento por defecto)."""
        print("[INFO] --rebuild: se borra la coleccion, el parent store y el manifest.")
        self.db_manager.recreate_collection()
        self.parent_store._data = {}
        self.manifest._data = {}
        self.parent_store.save()
        self.manifest.save()
        self.run()

    def _process_in_batches(self, chunks: List[Dict[str, Any]]) -> None:
        bs = self.config.batch_size
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            texts = [c["content"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            ids = [c["id"] for c in batch]
            vectors = self.embeddings_provider.embed_documents(texts)
            self.db_manager.upsert_batch(
                embeddings=vectors, documents=texts, metadatas=metadatas, ids=ids
            )
            print(f"      {min(i + bs, len(chunks))}/{len(chunks)} children vectorizados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Ingesta incremental: Contextual Retrieval + Parent-Child"
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="Elimina de la base los archivos que ya no existen en las carpetas fuente.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Borra TODO (coleccion, parent store, manifest) y reingesta desde cero.",
    )
    args = parser.parse_args()

    print("[START] Pipeline de Ingesta incremental: Contextual Retrieval + Parent-Child")
    config = IngestionConfig()
    pipeline = IngestionPipeline(config)
    if args.rebuild:
        pipeline.rebuild()
    else:
        pipeline.run(prune=args.prune)
