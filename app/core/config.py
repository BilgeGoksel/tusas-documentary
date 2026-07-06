"""Application settings."""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    ollama_chat_timeout_seconds: float = Field(default=120.0, gt=0.0)
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    indexed_dir: str = "data/indexed"
    max_upload_size_mb: int = 20
    pdf_ocr_min_text_chars: int = 30
    ocr_lang: str = "en"
    ocr_device: str = "cpu"
    ocr_language: str = "en"
    ocr_min_image_width: int = 32
    ocr_min_image_height: int = 32
    ocr_low_confidence_threshold: float = 0.50
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "document_chunks"
    retrieval_min_score: float = Field(default=0.15, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Ensure chunk overlap cannot consume a complete chunk."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE degerinden kucuk olmalidir.")
        return self

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
