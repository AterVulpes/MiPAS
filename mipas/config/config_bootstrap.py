import json
from pathlib import Path
from mipas.config.config_registry import CONFIG_REGISTRY

def get_default_config_for_analysis(analysis_key: str) -> dict:
    schema = CONFIG_REGISTRY.get(analysis_key, {})
    return {
        key: meta.get("default", "") for key, meta in schema.items()
    }

def initialize_config_file_if_missing(analysis_key: str, file_path: Path):
    if not file_path.exists():
        default_config = get_default_config_for_analysis(analysis_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return True  # File was created
    return False  # Already exists
