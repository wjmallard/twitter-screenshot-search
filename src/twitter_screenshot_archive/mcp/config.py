"""MCP-specific configuration from config.yaml."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path("config.yaml")

try:
    with open(_CONFIG_PATH) as f:
        _raw = yaml.safe_load(f) or {}
except FileNotFoundError:
    _raw = {}

LMSTUDIO_URL = _raw.get("lmstudio_url", "http://localhost:1234")
EMBEDDING_MODEL = _raw.get("embedding_model", "text-embedding-embeddinggemma-300m")
EMBEDDING_DIM = 768
BACKFILL_BATCH_SIZE = _raw.get("embedding_batch_size", 64)
DEFAULT_SEARCH_LIMIT = _raw.get("embedding_search_limit", 10)
SEARCH_SIMILARITY_FLOOR = _raw.get("search_similarity_floor", 0.3)
SNIPPET_MAX_CHARS = 300
