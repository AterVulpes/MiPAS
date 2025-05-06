import importlib
import pkgutil
import os

__path__ = [os.path.dirname(__file__)]

CONFIG_REGISTRY = {}

def _load_module_configs():
    import mipas.config
    prefix = __name__.rsplit('.', 1)[0] + "."
    for loader, name, is_pkg in pkgutil.iter_modules(__path__):
        if name.startswith("config_"):
            mod = importlib.import_module(prefix + name)
            key = name.replace("config_", "")
            if hasattr(mod, "CONFIG"):
                CONFIG_REGISTRY[key] = mod.CONFIG

_load_module_configs()
