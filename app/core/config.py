"""Application settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    max_upload_size_mb: int = 20
    pdf_ocr_min_text_chars: int = 30
    ocr_lang: str = "en"
    ocr_device: str = "cpu"
    ocr_language: str = "en"
    ocr_min_image_width: int = 32
    ocr_min_image_height: int = 32
    ocr_low_confidence_threshold: float = 0.50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
