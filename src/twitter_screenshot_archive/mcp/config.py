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
SNIPPET_MAX_CHARS = _raw.get("snippet_max_chars_mcp", 500)

# Clustering
PCA_N_COMPONENTS = _raw.get("pca_n_components", 15)
TIME_WEIGHT = _raw.get("time_weight", 2.0)
CLUSTER_MIN_SIZE = _raw.get("cluster_min_size", 3)
CLUSTER_MIN_SAMPLES = _raw.get("cluster_min_samples", None)  # defaults to CLUSTER_MIN_SIZE
TOPIC_SIM_THRESHOLD_PCT = _raw.get("topic_sim_threshold_pct", 0.30)
COARSE_SIMILARITY_FLOOR = _raw.get("coarse_sim_floor", 0.15)  # SQL pre-filter: deliberately loose
