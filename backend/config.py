"""
config.py — Central configuration for the Noir Whisper backend.
Model architecture constants and server settings live here.
"""

import os


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ModelConfig:
    VOCAB_SIZE: int   = 50257
    D_MODEL:    int   = 768
    N_LAYERS:   int   = 12
    N_HEADS:    int   = 12
    D_FF:       int   = 3072
    MAX_SEQ_LEN: int  = 1024
    DROPOUT:    float = 0.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = os.environ.get(
    "CHECKPOINT_PATH",
    "model/llm_model.pt"
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

SERVER_HOST: str = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT: int = int(os.environ.get("PORT", "8000"))

# Allowed CORS origins — frontend dev server + production placeholder
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite default
    "http://localhost:3000",   # CRA / alternative
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
]

# ---------------------------------------------------------------------------
# Generation defaults (used as fallback when client omits a field)
# ---------------------------------------------------------------------------

class GenerationDefaults:
    MAX_TOKENS:         int   = 512
    TEMPERATURE:        float = 0.7
    TOP_P:              float = 0.9
    TOP_K:              int   = 50
    REPETITION_PENALTY: float = 1.1
    RANGE_EPSILON:      float = 0.1