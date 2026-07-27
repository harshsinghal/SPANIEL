"""Compatibility helpers for tokenizer configs saved by newer Transformers."""

import json
from pathlib import Path


def tokenizer_load_kwargs(model_path):
    """Return safe overrides for a local tokenizer configuration.

    Transformers 5 serializes ``extra_special_tokens`` as a list, while the
    late 4.x loader expects a name-to-token mapping. Supplying the mapping is
    accepted by both versions and preserves which ids are treated as special.
    """
    config_path = Path(model_path) / "tokenizer_config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return {}
    return {
        "extra_special_tokens": {
            f"extra_special_token_{index}": token
            for index, token in enumerate(extra)
        }
    }
