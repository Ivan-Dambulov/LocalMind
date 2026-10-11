"""Central configuration for LocalMind."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    vectorstore_dir: Path = VECTORSTORE_DIR

    chunk_size: int = 512
    chunk_overlap: int = 64

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 5
    hybrid_alpha: float = 0.6
    retrieval_min_score: float = 0.35
    weak_local_action: str = "web"

    llm_model_path: str = ""
    llm_context_size: int = 4096
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    # auto | cuda | metal | rocm | cpu
    llm_preferred_backend: str = "auto"
    # -1 auto layers, 0 CPU, >0 fixed layer count
    llm_gpu_layers: int = -1

    max_agent_steps: int = 5


settings = Settings()
