"""Streamlit frontend for the local document analysis system."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
import streamlit as st

REQUEST_TIMEOUT_SECONDS = 10
PROCESS_REQUEST_TIMEOUT_SECONDS = 300
QA_REQUEST_TIMEOUT_SECONDS = 180
DEFAULT_BACKEND_BASE_URL = "http://localhost:8000"
DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
SUPPORTED_FILE_TYPES = ["pdf", "jpg", "jpeg", "png"]
UPLOADED_DOCUMENTS_KEY = "uploaded_documents"
PROCESSED_DOCUMENTS_KEY = "processed_documents"
INDEXED_DOCUMENTS_KEY = "indexed_documents"
PREPARED_DOCUMENTS_KEY = "prepared_documents"
CHAT_HISTORY_KEY = "qa_chat_history"
SELECTED_DOCUMENT_IDS_KEY = "qa_selected_document_ids"
QA_SELECTION_INITIALIZED_KEY = "qa_selection_initialized"
QA_TOP_K_KEY = "qa_top_k"
TEXT_PREVIEW_LIMIT = 1000


def read_dotenv() -> dict[str, str]:
    """Read simple KEY=VALUE pairs from a local .env file."""
    env_path = Path(".env")
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
            continue
        key, value = clean_line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_setting(name: str, default: str, dotenv_values: dict[str, str]) -> str:
    """Return a setting from environment variables, .env, or a default."""
    return os.getenv(name) or dotenv_values.get(name) or default


def get_backend_base_url(dotenv_values: dict[str, str]) -> str:
    """Return the configured backend base URL without a trailing slash."""
    return get_setting("BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL, dotenv_values).rstrip("/")


def get_health_status(backend_base_url: str) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch backend health status without raising UI-breaking errors."""
    try:
        response = requests.get(
            f"{backend_base_url}/api/v1/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json(), None
        return None, extract_error_message(response)
    except requests.RequestException:
        return None, "Backend'e erisilemiyor."


def extract_error_message(response: requests.Response) -> str:
    """Extract a readable error message from a backend response."""
    try:
        payload = response.json()
    except ValueError:
        return f"Istek basarisiz oldu. HTTP {response.status_code}"

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    return f"Istek basarisiz oldu. HTTP {response.status_code}"


def upload_document(backend_base_url: str, file) -> tuple[dict[str, Any] | None, str | None]:
    """Upload a selected document to the backend."""
    try:
        files = {
            "file": (
                file.name,
                file.getvalue(),
                file.type or "application/octet-stream",
            )
        }
        response = requests.post(
            f"{backend_base_url}/api/v1/documents/upload",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json(), None
        return None, extract_error_message(response)
    except requests.RequestException:
        return None, "Backend'e erisilemiyor."


def process_uploaded_document(
    backend_base_url: str,
    document_id: str,
    force: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Process an uploaded document by id through the backend."""
    try:
        response = requests.post(
            f"{backend_base_url}/api/v1/documents/{document_id}/process",
            params={"force": "true"} if force else None,
            timeout=PROCESS_REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json(), None
        return None, extract_error_message(response)
    except requests.RequestException:
        return None, "Backend'e erisilemiyor."


def index_processed_document(
    backend_base_url: str,
    document_id: str,
    force: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Index a processed document through the backend."""
    try:
        response = requests.post(
            f"{backend_base_url}/api/v1/documents/{document_id}/index",
            params={"force": "true"} if force else None,
            timeout=PROCESS_REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json(), None
        return None, extract_error_message(response)
    except requests.RequestException:
        return None, "Backend'e erisilemiyor."
    except ValueError:
        return None, "Backend gecersiz bir yanit dondurdu."


def ask_qa(
    backend_base_url: str,
    query: str,
    document_ids: list[str],
    top_k: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send a grounded question to the backend QA endpoint."""
    try:
        response = requests.post(
            f"{backend_base_url}/api/v1/qa",
            json={
                "query": query,
                "document_ids": document_ids,
                "top_k": top_k,
            },
            timeout=QA_REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json(), None
        return None, extract_error_message(response)
    except requests.RequestException:
        return None, "Backend'e erisilemiyor."
    except ValueError:
        return None, "Backend gecersiz bir yanit dondurdu."


def process_document(
    backend_base_url: str,
    document_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Process an uploaded document using the standard cached flow."""
    return process_uploaded_document(backend_base_url, document_id, force=False)


def index_document(
    backend_base_url: str,
    document_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Make a processed document searchable using the standard cached flow."""
    return index_processed_document(backend_base_url, document_id, force=False)


def ask_question(
    backend_base_url: str,
    query: str,
    document_ids: list[str],
    top_k: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ask a grounded question about prepared documents."""
    return ask_qa(backend_base_url, query, document_ids, top_k)


def prepare_document_pipeline(
    backend_base_url: str,
    file: Any,
    on_step: Callable[[str, str, float], None] | None = None,
) -> dict[str, Any]:
    """Upload, process and prepare one document while preserving stage errors."""
    _report_pipeline_step(on_step, "upload", "Belge yükleniyor...", 0.10)
    upload_payload, error = upload_document(backend_base_url, file)
    if error is not None or upload_payload is None:
        return _pipeline_failure(file.name, "upload", error)

    document_id = str(upload_payload.get("document_id", ""))
    if not document_id:
        return _pipeline_failure(file.name, "upload", "Belge kimliği alınamadı.")

    _report_pipeline_step(
        on_step,
        "process",
        "Metin çıkarılıyor / OCR uygulanıyor...",
        0.45,
    )
    process_payload, error = process_document(backend_base_url, document_id)
    if error is not None or process_payload is None:
        return _pipeline_failure(file.name, "process", error, document_id)

    _report_pipeline_step(
        on_step,
        "index",
        "Belge aranabilir hale getiriliyor...",
        0.75,
    )
    index_payload, error = index_document(backend_base_url, document_id)
    if error is not None or index_payload is None:
        return _pipeline_failure(file.name, "index", error, document_id)

    _report_pipeline_step(on_step, "ready", "Hazır.", 1.0)
    return {
        "prepared": True,
        "document_id": document_id,
        "original_filename": str(
            index_payload.get("original_filename")
            or upload_payload.get("original_filename")
            or file.name
        ),
        "upload_status": str(upload_payload.get("status", "uploaded")),
        "processing_status": str(
            process_payload.get("processing_status", "processed")
        ),
        "indexing_status": str(index_payload.get("indexing_status", "indexed")),
        "page_count": int(process_payload.get("page_count", 0)),
        "chunk_count": int(index_payload.get("chunk_count", 0)),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "from_cache": bool(
            process_payload.get("from_cache") or index_payload.get("from_cache")
        ),
        "indexed_at": index_payload.get("indexed_at"),
    }


def _report_pipeline_step(
    callback: Callable[[str, str, float], None] | None,
    stage: str,
    message: str,
    progress: float,
) -> None:
    if callback is not None:
        callback(stage, message, progress)


def _pipeline_failure(
    filename: str,
    stage: str,
    error: str | None,
    document_id: str | None = None,
) -> dict[str, Any]:
    return {
        "prepared": False,
        "original_filename": filename,
        "document_id": document_id,
        "failed_stage": stage,
        "error": error or "Beklenmeyen bir hata oluştu.",
    }


def upload_documents(
    backend_base_url: str,
    files: list[Any],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Upload selected documents one by one and return per-file results."""
    results: list[dict[str, Any]] = []
    total_files = len(files)
    for index, file in enumerate(files, start=1):
        payload, error = upload_document(backend_base_url, file)
        results.append(
            {
                "filename": file.name,
                "payload": payload,
                "error": error,
            }
        )
        if on_progress is not None:
            on_progress(index, total_files)
    return results


def initialize_session_state() -> None:
    """Initialize Streamlit session state keys used by the app."""
    if UPLOADED_DOCUMENTS_KEY not in st.session_state:
        st.session_state[UPLOADED_DOCUMENTS_KEY] = []
    if PROCESSED_DOCUMENTS_KEY not in st.session_state:
        st.session_state[PROCESSED_DOCUMENTS_KEY] = {}
    if INDEXED_DOCUMENTS_KEY not in st.session_state:
        st.session_state[INDEXED_DOCUMENTS_KEY] = {}
    if PREPARED_DOCUMENTS_KEY not in st.session_state:
        st.session_state[PREPARED_DOCUMENTS_KEY] = {}
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []
    if SELECTED_DOCUMENT_IDS_KEY not in st.session_state:
        st.session_state[SELECTED_DOCUMENT_IDS_KEY] = []
    if QA_SELECTION_INITIALIZED_KEY not in st.session_state:
        st.session_state[QA_SELECTION_INITIALIZED_KEY] = False


def remember_uploaded_document(document: dict[str, Any]) -> None:
    """Append a document to session state unless its document_id already exists."""
    document_id = document.get("document_id")
    existing_ids = {
        item.get("document_id")
        for item in st.session_state[UPLOADED_DOCUMENTS_KEY]
    }
    if document_id not in existing_ids:
        st.session_state[UPLOADED_DOCUMENTS_KEY].append(document)


def format_size(size_bytes: int) -> str:
    """Format bytes for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_sidebar(dotenv_values: dict[str, str], backend_base_url: str) -> None:
    """Render backend, Ollama, and model status in the sidebar."""
    st.sidebar.header("Durum")
    st.sidebar.caption(f"Backend: {backend_base_url}")

    health_payload, health_error = get_health_status(backend_base_url)
    if health_payload is None:
        st.sidebar.error("Backend unavailable")
        st.sidebar.caption(health_error)
        ollama_status = "unknown"
    else:
        api_status = health_payload.get("api_status", "unknown")
        ollama_status = health_payload.get("ollama", {}).get("status", "unknown")
        st.sidebar.success(f"API: {api_status}")
        if ollama_status == "available":
            st.sidebar.success("Ollama: available")
        else:
            st.sidebar.warning(f"Ollama: {ollama_status}")

    st.sidebar.header("Modeller")
    st.sidebar.write(
        "Chat:",
        get_setting("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL, dotenv_values),
    )
    st.sidebar.write(
        "Embedding:",
        get_setting("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL, dotenv_values),
    )
    st.sidebar.header("Soru-Cevap")
    st.sidebar.slider(
        "Getirilecek kaynak sayisi",
        min_value=1,
        max_value=10,
        value=5,
        key=QA_TOP_K_KEY,
    )


def render_upload_area(backend_base_url: str) -> None:
    """Render the one-click document preparation workflow."""
    st.header("1. Belge Yükle ve Hazırla")
    uploaded_files = st.file_uploader(
        "Belgeleri seç",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    render_selected_files(uploaded_files)

    if st.button("Belgeleri Yükle ve Hazırla", type="primary"):
        if not uploaded_files:
            st.warning("Lütfen hazırlamak için en az bir belge seçin.")
            return

        for file in uploaded_files:
            status_box = st.status(
                f"{file.name} hazırlanıyor...",
                expanded=True,
            )
            progress_bar = status_box.progress(0)

            def update_step(stage: str, message: str, progress: float) -> None:
                del stage
                status_box.write(message)
                progress_bar.progress(progress)

            result = prepare_document_pipeline(
                backend_base_url,
                file,
                on_step=update_step,
            )
            if result.get("prepared") is True:
                remember_prepared_document(result)
                status_box.update(
                    label=f"{file.name} — Hazır.",
                    state="complete",
                    expanded=False,
                )
                continue

            stage_labels = {
                "upload": "Belge yüklenemedi",
                "process": "Metin çıkarılamadı veya OCR tamamlanamadı",
                "index": "Belge aranabilir hale getirilemedi",
            }
            stage_label = stage_labels.get(
                str(result.get("failed_stage")), "Belge hazırlanamadı"
            )
            status_box.write(f"{stage_label}: {result.get('error')}")
            status_box.update(
                label=f"{file.name} — Hazırlanamadı.",
                state="error",
                expanded=True,
            )

    render_prepared_documents()


def remember_prepared_document(document: dict[str, Any]) -> None:
    """Store one prepared document without duplicating its document id."""
    document_id = str(document.get("document_id", ""))
    if not document_id:
        return
    prepared_documents = st.session_state[PREPARED_DOCUMENTS_KEY]
    existing_ids = set(prepared_documents)
    selected_ids = set(st.session_state[SELECTED_DOCUMENT_IDS_KEY])
    selected_all_existing = bool(existing_ids) and selected_ids == existing_ids
    prepared_documents[document_id] = document
    if selected_all_existing and document_id not in selected_ids:
        st.session_state[SELECTED_DOCUMENT_IDS_KEY].append(document_id)


def render_prepared_documents() -> None:
    """Render user-friendly prepared documents with optional technical details."""
    st.header("2. Hazır Belgeler")
    prepared_documents = st.session_state[PREPARED_DOCUMENTS_KEY]
    if not prepared_documents:
        st.info("Henüz soru sormaya hazır bir belge yok.")
        return

    for document in prepared_documents.values():
        filename = str(document.get("original_filename", "Belge"))
        st.success(f"✓ {filename} — Soru sormaya hazır")
        with st.expander(f"{filename} teknik detayları", expanded=False):
            st.write("document_id:", document.get("document_id"))
            st.write("page_count:", document.get("page_count", 0))
            st.write("chunk_count:", document.get("chunk_count", 0))
            st.write("from_cache:", document.get("from_cache", False))
            st.write("indexed_at:", document.get("indexed_at"))
            st.write("prepared_at:", document.get("prepared_at"))


def render_selected_files(uploaded_files: list[Any]) -> None:
    """Render selected files before upload."""
    if not uploaded_files:
        st.info("PDF, JPG, JPEG veya PNG belgeleri secin.")
        return

    st.subheader("Secilen Belgeler")
    for uploaded_file in uploaded_files:
        with st.expander(uploaded_file.name, expanded=False):
            st.write("Dosya adi:", uploaded_file.name)
            st.write("MIME turu:", uploaded_file.type or "bilinmiyor")
            st.write("Boyut:", format_size(uploaded_file.size))


def render_upload_results(results: list[dict[str, Any]]) -> None:
    """Render per-file upload results for the latest upload attempt."""
    st.subheader("Yukleme Sonuclari")
    for result in results:
        filename = result["filename"]
        payload = result["payload"]
        error = result["error"]
        if error is not None:
            st.error(f"{filename}: {error}")
            continue
        if payload is None:
            st.error(f"{filename}: Beklenmeyen bir hata olustu.")
            continue
        if payload.get("is_duplicate") is True:
            st.warning(
                f"{filename}: Bu belge daha once yuklenmis. "
                f"document_id={payload.get('document_id')}"
            )
            continue
        st.success(f"{filename}: belge yuklendi. document_id={payload.get('document_id')}")


def render_uploaded_documents(backend_base_url: str) -> None:
    """Render successfully uploaded documents stored in session state."""
    st.subheader("Yuklenen Belgeler")
    st.caption(
        "Listeyi Temizle yalnizca bu ekrandaki oturum listesini temizler; "
        "diskteki dosyalari silmez."
    )
    if st.button("Listeyi Temizle"):
        st.session_state[UPLOADED_DOCUMENTS_KEY] = []
        st.info("Goruntulenen belge listesi temizlendi. Diskteki dosyalar silinmedi.")
        return

    uploaded_documents = st.session_state[UPLOADED_DOCUMENTS_KEY]
    if not uploaded_documents:
        st.info("Bu oturumda henuz basarili belge yuklenmedi.")
        return

    for document in uploaded_documents:
        title = document.get("original_filename", "Belge")
        with st.expander(title, expanded=False):
            render_uploaded_document_details(document)
            render_processing_controls(backend_base_url, document)
            render_processing_result(document)


def render_uploaded_document_details(document: dict[str, Any]) -> None:
    """Render stored upload metadata for one document."""
    st.write("Orijinal dosya adi:", document.get("original_filename"))
    st.write("document_id:", document.get("document_id"))
    st.write("MIME turu:", document.get("content_type"))
    st.write("Boyut:", format_size(int(document.get("size_bytes", 0))))
    st.write("Durum:", document.get("status"))
    st.write("Olusturulma zamani:", document.get("created_at"))


def render_processing_controls(backend_base_url: str, document: dict[str, Any]) -> None:
    """Render process and force reprocess buttons for one document."""
    document_id = str(document.get("document_id", ""))
    if not document_id:
        st.warning("Belge kimligi bulunamadi.")
        return

    processed_documents = st.session_state[PROCESSED_DOCUMENTS_KEY]
    is_processed = document_id in processed_documents
    if is_processed:
        st.success("Belge islendi.")
    else:
        st.info("Bu belge henuz islenmedi.")

    process_col, reprocess_col = st.columns(2)
    with process_col:
        if st.button("Belgeyi Isle", key=f"process_{document_id}", disabled=is_processed):
            process_document_from_ui(backend_base_url, document_id, force=False)
    with reprocess_col:
        if st.button("Yeniden Isle", key=f"reprocess_{document_id}"):
            process_document_from_ui(backend_base_url, document_id, force=True)
    st.caption("Yeniden Isle cache sonucunu atlar ve belgeyi tekrar isler.")

    if is_processed:
        render_indexing_controls(backend_base_url, document)


def render_indexing_controls(
    backend_base_url: str,
    document: dict[str, Any],
) -> None:
    """Render indexing controls for one processed document."""
    document_id = str(document.get("document_id", ""))
    indexed_documents = st.session_state[INDEXED_DOCUMENTS_KEY]
    is_indexed = document_id in indexed_documents
    if is_indexed:
        st.success("Belge indexlendi ve soru-cevap icin hazir.")
    else:
        st.info("Belge henuz indexlenmedi.")

    index_col, reindex_col = st.columns(2)
    with index_col:
        if st.button(
            "Belgeyi Indexle",
            key=f"index_{document_id}",
            disabled=is_indexed,
        ):
            index_document_from_ui(backend_base_url, document, force=False)
    with reindex_col:
        if st.button("Yeniden Indexle", key=f"reindex_{document_id}"):
            index_document_from_ui(backend_base_url, document, force=True)


def index_document_from_ui(
    backend_base_url: str,
    document: dict[str, Any],
    force: bool,
) -> None:
    """Call the indexing endpoint and remember successful index metadata."""
    document_id = str(document.get("document_id", ""))
    spinner_text = "Belge yeniden indexleniyor..." if force else "Belge indexleniyor..."
    with st.spinner(spinner_text):
        payload, error = index_processed_document(
            backend_base_url,
            document_id=document_id,
            force=force,
        )
    if error is not None:
        st.error(f"Belge indexlenemedi: {error}")
        return
    if payload is None:
        st.error("Belge indexlenemedi: Beklenmeyen bir hata olustu.")
        return

    indexed_record = dict(payload)
    indexed_record["original_filename"] = document.get(
        "original_filename", payload.get("original_filename", "Belge")
    )
    st.session_state[INDEXED_DOCUMENTS_KEY][document_id] = indexed_record
    st.success("Belge indexleme tamamlandi.")


def process_document_from_ui(
    backend_base_url: str,
    document_id: str,
    force: bool,
) -> None:
    """Call the processing endpoint and store the result in session state."""
    spinner_text = "Belge yeniden isleniyor..." if force else "Belge isleniyor..."
    with st.spinner(spinner_text):
        payload, error = process_uploaded_document(
            backend_base_url,
            document_id=document_id,
            force=force,
        )
    if error is not None:
        st.error(f"Belge islenemedi: {error}")
        return
    if payload is None:
        st.error("Belge islenemedi: Beklenmeyen bir hata olustu.")
        return

    st.session_state[PROCESSED_DOCUMENTS_KEY][document_id] = payload
    st.success("Belge isleme tamamlandi.")


def render_processing_result(document: dict[str, Any]) -> None:
    """Render cached processing result for a document if present."""
    document_id = str(document.get("document_id", ""))
    result = st.session_state[PROCESSED_DOCUMENTS_KEY].get(document_id)
    if result is None:
        return

    st.divider()
    st.subheader("Isleme Sonucu")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Durum", str(result.get("processing_status", "bilinmiyor")))
    metric_cols[1].metric("Sayfa", int(result.get("page_count", 0)))
    metric_cols[2].metric("Karakter", int(result.get("total_character_count", 0)))
    cache_label = "Evet" if result.get("from_cache") else "Hayir"
    metric_cols[3].metric("Cache", cache_label)
    st.write("Islenme zamani:", result.get("processed_at"))

    render_page_previews(result.get("pages", []), document_id)


def render_page_previews(pages: list[dict[str, Any]], document_id: str) -> None:
    """Render page-level extraction previews."""
    if not pages:
        st.warning("Isleme sonucu sayfa bilgisi icermiyor.")
        return

    for page in pages:
        page_number = page.get("page_number", "?")
        method_label = format_extraction_method(str(page.get("extraction_method", "")))
        title = f"Sayfa {page_number} - {method_label}"
        with st.expander(title, expanded=False):
            st.write("Sayfa numarasi:", page_number)
            st.write("Extraction method:", method_label)
            st.write("Karakter sayisi:", page.get("character_count", 0))
            confidence = page.get("confidence")
            if confidence is not None:
                st.write("OCR confidence:", f"{float(confidence):.2f}")
            warnings = page.get("warnings") or []
            for warning in warnings:
                st.warning(str(warning))

            text = str(page.get("text") or "")
            preview = text[:TEXT_PREVIEW_LIMIT]
            if len(text) > TEXT_PREVIEW_LIMIT:
                preview = f"{preview}\n\n..."
            st.text_area(
                "Ilk 1000 karakter",
                value=preview,
                height=180,
                disabled=True,
                key=f"preview_{document_id}_{page_number}",
            )
            if len(text) > TEXT_PREVIEW_LIMIT:
                with st.expander("Metnin tamami", expanded=False):
                    st.text_area(
                        "Tam metin",
                        value=text,
                        height=300,
                        disabled=True,
                        key=f"full_text_{document_id}_{page_number}",
                    )


def format_extraction_method(extraction_method: str) -> str:
    """Return a user-facing extraction method label."""
    labels = {
        "native_pdf": "Native PDF",
        "ocr": "OCR",
        "ocr_pdf": "PDF OCR",
    }
    return labels.get(extraction_method, extraction_method or "bilinmiyor")


def render_qa_area(backend_base_url: str) -> None:
    """Render prepared-document selection and persistent QA chat history."""
    st.divider()
    st.header("3. Belgelere Soru Sor")

    if st.button("Sohbeti Temizle", key="clear_qa_chat"):
        st.session_state[CHAT_HISTORY_KEY] = []
        st.success("Sohbet gecmisi temizlendi.")

    prepared_documents = st.session_state[PREPARED_DOCUMENTS_KEY]
    if not prepared_documents:
        st.info(
            "Soru-cevap alanını kullanmak için önce bir belgeyi hazırlayın."
        )
        render_chat_history()
        return

    available_ids = list(prepared_documents)
    if not st.session_state[QA_SELECTION_INITIALIZED_KEY]:
        st.session_state[SELECTED_DOCUMENT_IDS_KEY] = available_ids
        st.session_state[QA_SELECTION_INITIALIZED_KEY] = True
    valid_selection = [
        document_id
        for document_id in st.session_state[SELECTED_DOCUMENT_IDS_KEY]
        if document_id in prepared_documents
    ]
    st.session_state[SELECTED_DOCUMENT_IDS_KEY] = valid_selection
    selected_ids = st.multiselect(
        "Soru sorulacak belgeler",
        options=available_ids,
        format_func=lambda document_id: str(
            prepared_documents[document_id].get("original_filename", "Belge")
        ),
        key=SELECTED_DOCUMENT_IDS_KEY,
        help="Varsayılan olarak tüm hazır belgelerde arama yapılır.",
    )

    render_chat_history()
    query = st.chat_input("Belgeleriniz hakkinda bir soru sorun")
    if query is None:
        return
    if not selected_ids:
        st.warning("Soru sormak için en az bir hazır belge seçin.")
        return

    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Belgelerde yanit araniyor..."):
        payload, error = ask_question(
            backend_base_url,
            query=query,
            document_ids=list(selected_ids),
            top_k=int(st.session_state.get(QA_TOP_K_KEY, 5)),
        )
    if error is not None:
        with st.chat_message("assistant"):
            st.error(error)
        return
    if payload is None:
        with st.chat_message("assistant"):
            st.error("Cevap alinamadi: Beklenmeyen bir hata olustu.")
        return

    exchange = {
        "query": query,
        "answer": str(payload.get("answer", "")),
        "sources": payload.get("sources") or [],
        "document_ids": list(selected_ids),
        "found_in_documents": bool(payload.get("found_in_documents")),
        "retrieved_chunk_count": int(payload.get("retrieved_chunk_count", 0)),
        "model": str(payload.get("model", "")),
        "top_k": int(payload.get("top_k", st.session_state.get(QA_TOP_K_KEY, 5))),
    }
    st.session_state[CHAT_HISTORY_KEY].append(exchange)
    render_assistant_exchange(exchange)


def render_chat_history() -> None:
    """Render all stored question-answer exchanges in chronological order."""
    for exchange in st.session_state[CHAT_HISTORY_KEY]:
        with st.chat_message("user"):
            st.markdown(str(exchange.get("query", "")))
        render_assistant_exchange(exchange)


def render_assistant_exchange(exchange: dict[str, Any]) -> None:
    """Render one assistant answer and its user-facing source details."""
    with st.chat_message("assistant"):
        answer = str(exchange.get("answer", ""))
        if exchange.get("found_in_documents") is False:
            st.warning(answer)
        else:
            st.markdown(answer)
        render_sources(exchange.get("sources") or [])


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render source metadata without exposing internal document identifiers."""
    if not sources:
        return
    with st.expander("Kaynaklar", expanded=False):
        for source in sources:
            source_number = source.get("source_number", "?")
            st.markdown(
                f"**[{source_number}] {source.get('original_filename', 'Belge')}**"
            )
            st.write("Sayfa:", source.get("page_number", "?"))
            score = source.get("similarity_score")
            if isinstance(score, (int, float)):
                st.write("Benzerlik skoru:", f"{float(score):.3f}")
            snippet = str(source.get("snippet") or "").strip()
            if snippet:
                st.caption(snippet)
            st.divider()


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Yerel Belge Analiz ve Soru-Cevap Sistemi",
        layout="centered",
    )
    initialize_session_state()
    dotenv_values = read_dotenv()
    backend_base_url = get_backend_base_url(dotenv_values)

    st.title("Yerel Belge Analiz ve Soru-Cevap Sistemi")
    st.write("Sistem PDF, JPG ve PNG belgelerini yerel olarak isleyecektir.")

    render_sidebar(dotenv_values, backend_base_url)
    render_upload_area(backend_base_url)
    render_qa_area(backend_base_url)


if __name__ == "__main__":
    main()
