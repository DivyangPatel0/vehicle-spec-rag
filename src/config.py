"""Configuration settings for the Vehicle Specification Extraction system."""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # File Paths
    project_root: Path = PROJECT_ROOT
    pdf_path: Path = PROJECT_ROOT / "sample-service-manual 1.pdf"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    output_dir: Path = PROJECT_ROOT / "outputs"
    deliverables_dir: Path = PROJECT_ROOT / "deliverables"
    
    # Chunking Configuration
    chunk_size: int = 800          # characters per chunk target
    chunk_overlap: int = 150       # overlap between adjacent chunks
    preserve_tables: bool = True   # keep multi-line table specs intact
    
    # Retrieval Configuration
    top_k: int = 5                 # number of chunks to retrieve per query
    similarity_threshold: float = 0.25 # minimum similarity score
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock") # "mock", "openai", "gemini"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.0
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deliverables_dir.mkdir(parents=True, exist_ok=True)

# Default Global Config Instance
default_config = Config()
