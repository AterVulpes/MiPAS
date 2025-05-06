# mipas/config/config_manager.py

from mipas.config.config_registry import CONFIG_REGISTRY

# Mapping short analysis keys to internal schema keys
ANALYSIS_KEY_MAP = {
    "linear": "linear_entropy_analysis",
    "segmented": "segmented_entropy_analysis",
    "fd-box": "fd_box_analysis",
    "fd-wtmm": "fd_wtmm_analysis",
}

class ConfigSchema:
    def __init__(self, user_config: dict, analysis_type: str):
        resolved_key = ANALYSIS_KEY_MAP.get(analysis_type, analysis_type)

        if resolved_key not in CONFIG_REGISTRY:
            raise ValueError(f"Unknown analysis type: '{analysis_type}'")

        self.analysis_type = resolved_key
        self.schema = CONFIG_REGISTRY[resolved_key]
        self.data = {}

        for key, meta in self.schema.items():
            raw_value = user_config.get(key, meta.get("default"))
            self.data[key] = self._validate_or_default(key, raw_value, meta)

    def _validate_or_default(self, key, value, meta):
        try:
            if meta["validate"](value):
                return value
        except Exception:
            pass
        return meta.get("default")

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, fallback=None):
        return self.data.get(key, fallback)

    def __contains__(self, key):
        return key in self.data

    def to_dict(self):
        return self.data.copy()

    def as_schema_fields(self):
        """
        Return a GUI-friendly structure: {key: {"type", "label", "hint"}}.
        """
        return {
            key: {
                "type": meta["type"],
                "label": meta.get("label", key),
                "hint": meta.get("hint", "")
            }
            for key, meta in self.schema.items()
        }
