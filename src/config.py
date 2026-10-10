"""Central configuration for LocalMind."""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


class Settings(BaseSettings):
    # Paths
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    vectorstore_dir: Path = VECTORSTORE_DIR

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatives: "BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2"

    # Retrieval
    top_k: int = 5
    hybrid_alpha: float = 0.6  # weight for dense vs BM25 (0=only BM25, 1=only dense)

    # LLM
    llm_model_path: str = ""  # path to .gguf file; leave empty to use mock / later cloud
    llm_context_size: int = 4096
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Agent
    max_agent_steps: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
