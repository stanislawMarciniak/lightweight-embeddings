import os
from functools import lru_cache
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core
    SUPABASE_URL: AnyHttpUrl = Field(..., env="SUPABASE_URL")
    SUPABASE_KEY: str = Field(..., env="SUPABASE_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    # Optional: CORS / frontend (single origin or comma-separated list)
    FRONTEND_ORIGIN: str | None = Field(default=None, env="FRONTEND_ORIGIN")

    # Storage
    SUPABASE_STORAGE_BUCKET: str = Field(default="documents", env="SUPABASE_STORAGE_BUCKET")

    # Embeddings / similarity
    GLOVE_PATH: str = Field(
        default=os.path.join(os.path.dirname(__file__), "models", "glove.6B.100d.txt"),
        env="GLOVE_PATH",
    )
    # Custom_hybrid encoder weights (torch-free numpy export). 128-d output.
    SEMANTIC_MODEL_PATH: str = Field(
        default=os.path.join(os.path.dirname(__file__), "models", "custom_hybrid_encoder.npz"),
        env="SEMANTIC_MODEL_PATH",
    )
    # Frozen BERT token-embedding table (30522 x 768) for WordPiece lookup.
    BERT_EMBEDDINGS_PATH: str = Field(
        default=os.path.join(os.path.dirname(__file__), "models", "bert_token_embeddings.npz"),
        env="BERT_EMBEDDINGS_PATH",
    )
    # Path to the FP32 PyTorch checkpoint (used by export/finetune scripts).
    CUSTOM_HYBRID_PT_PATH: str = Field(
        default=os.path.join(os.path.dirname(__file__), "models", "custom_hybrid.pt"),
        env="CUSTOM_HYBRID_PT_PATH",
    )
    # Optimized ONNX export (used when EMBEDDING_RUNTIME=onnx).
    SEMANTIC_ONNX_PATH: str = Field(
        default=os.path.join(os.path.dirname(__file__), "models", "custom_hybrid_encoder.onnx"),
        env="SEMANTIC_ONNX_PATH",
    )
    # Inference runtime for the custom encoder: "numpy" (no deps) or "onnx" (onnxruntime).
    EMBEDDING_RUNTIME: str = Field(default="numpy", env="EMBEDDING_RUNTIME")
    # Which encoder powers embeddings: "custom" (CompactSimilarityModel, 128-d) or "glove" (mean, 100-d).
    EMBEDDING_BACKEND: str = Field(default="custom", env="EMBEDDING_BACKEND")

    # Semantic cache (in-process)
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.98, env="SEMANTIC_CACHE_THRESHOLD")
    SEMANTIC_CACHE_MAX_SIZE: int = Field(default=1024, env="SEMANTIC_CACHE_MAX_SIZE")
    SEMANTIC_CACHE_TTL_SECONDS: float = Field(default=3600.0, env="SEMANTIC_CACHE_TTL_SECONDS")
    FAQ_SIMILARITY_THRESHOLD: float = Field(default=0.95, env="FAQ_SIMILARITY_THRESHOLD")
    DOC_SIMILARITY_THRESHOLD: float = Field(default=0.80, env="DOC_SIMILARITY_THRESHOLD")

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()