"""Load config from config.json"""

import json
import os

import pandas as pd


class Config:
    """Simple config loader that allows attribute-style access."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    @classmethod
    def from_json(cls, filepath):
        """Load config from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

cfg = Config.from_json(CONFIG_PATH)

# Lazy loading des données - Ne charge que quand nécessaire
_recipe_cache = None
_recipe_rating_cache = None


def get_recipe_data():
    """
    Charge les données de recettes (avec cache).

    Returns:
        pd.DataFrame: Les données de recettes
    """
    global _recipe_cache
    if _recipe_cache is None:
        _recipe_cache = pd.read_csv(cfg.data_path)
    return _recipe_cache


def get_recipe_rating_data():
    """
    Charge les données de ratings (avec cache).

    Returns:
        pd.DataFrame: Les données de ratings
    """
    global _recipe_rating_cache
    if _recipe_rating_cache is None:
        _recipe_rating_cache = pd.read_csv(cfg.data_rating_path)
    return _recipe_rating_cache


# Ces proxies permettent d'accéder aux données de manière transparente
class _DataFrameProxy:
    """Proxy pour charger DataFrame uniquement quand accédé."""

    def __init__(self, loader_func):
        self._loader = loader_func
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __getitem__(self, key):
        return self._load()[key]

    def __len__(self):
        return len(self._load())

    def __repr__(self):
        return repr(self._load())

    def __str__(self):
        return str(self._load())


# Variables pour compatibilité avec l'ancien code
recipe = _DataFrameProxy(get_recipe_data)
recipe_rating = _DataFrameProxy(get_recipe_rating_data)
