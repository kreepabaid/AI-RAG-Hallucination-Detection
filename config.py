# ============================================================
#  CONFIG
# ============================================================

import os
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# HuggingFace API key - read from env or fallback to string
HF_API_KEY = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or ""

# Supported LLM models (runs on HuggingFace cloud via huggingface_hub SDK)
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)

AVAILABLE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

# Embedding model (runs on local CPU/GPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI verification model (runs on local CPU/GPU)
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Number of chunks to retrieve per question
TOP_K = 5