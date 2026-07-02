"""Streamlit frontend for the local document analysis system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import requests
import streamlit as st

REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_BACKEND_BASE_URL = "http://localhost:8000"
DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
SUPPORTED_FILE_TYPES = ["pdf", "jpg", "jpeg", "png"]
UPLOADED_DOCUMENTS_KEY = "uploaded_documents"


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
    except requests.RequestException as exc:
        return None, f"Backend'e erisilemiyor: {exc}"


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
    except requests.RequestException as exc:
        return None, f"Backend'e erisilemiyor: {exc}"


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


def render_upload_area(backend_base_url: str) -> None:
    """Render the document upload controls."""
    uploaded_files = st.file_uploader(
        "Belgeleri sec",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    render_selected_files(uploaded_files)

    if st.button("Secili Belgeleri Yukle", type="primary"):
        if not uploaded_files:
            st.warning("Lutfen yuklemek icin en az bir belge secin.")
            return

        with st.spinner("Secili belgeler yukleniyor..."):
            progress_bar = st.progress(0)
            results = upload_documents(
                backend_base_url,
                uploaded_files,
                on_progress=lambda current, total: progress_bar.progress(current / total),
            )

        for result in results:
            payload = result["payload"]
            error = result["error"]
            if payload is not None and error is None:
                remember_uploaded_document(payload)

        render_upload_results(results)

    render_uploaded_documents()


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


def render_uploaded_documents() -> None:
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
            st.write("Orijinal dosya adi:", document.get("original_filename"))
            st.write("document_id:", document.get("document_id"))
            st.write("MIME turu:", document.get("content_type"))
            st.write("Boyut:", format_size(int(document.get("size_bytes", 0))))
            st.write("Durum:", document.get("status"))
            st.write("Olusturulma zamani:", document.get("created_at"))


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


if __name__ == "__main__":
    main()
