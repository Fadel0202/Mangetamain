"""Load config from config.json"""

import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Importer streamlit pour le cache
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


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

# URL de la GitHub Release
GITHUB_RELEASE_URL = "https://github.com/Fadel0202/Mangetamain/releases/download/v1.0.0/data.zip"


def download_data_if_needed():
    """Télécharge les données depuis GitHub Release si elles n'existent pas."""
    data_dir = Path("data")
    recipe_file = data_dir / "RAW_recipes.csv"

    if recipe_file.exists():
        return

    print("Téléchargement des données...")

    try:
        data_dir.mkdir(exist_ok=True)
        response = requests.get(GITHUB_RELEASE_URL, timeout=300, stream=True)
        response.raise_for_status()

        zip_path = "data.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        os.remove(zip_path)
        print("Données téléchargées!")

    except Exception as e:
        print(f" Erreur: {e}")
        raise


# Télécharger les données au chargement du module
download_data_if_needed()

# Utiliser le cache Streamlit pour économiser la mémoire
if HAS_STREAMLIT:
    @st.cache_data
    def get_recipe_data():
        """Charge les données de recettes (avec cache Streamlit)."""
        # Charger seulement les colonnes nécessaires pour économiser la mémoire
        return pd.read_csv(
            cfg.data_path,
            dtype={
                'minutes': 'int32',
                'n_steps': 'int16',
                'n_ingredients': 'int16'
            }
        )

    @st.cache_data
    def get_recipe_rating_data():
        """Charge les données de ratings (avec cache Streamlit)."""
        return pd.read_csv(cfg.data_rating_path)
else:
    # Fallback sans Streamlit
    _recipe_cache = None
    _recipe_rating_cache = None

    def get_recipe_data():
        """Charge les données de recettes."""
        global _recipe_cache
        if _recipe_cache is None:
            _recipe_cache = pd.read_csv(cfg.data_path)
        return _recipe_cache

    def get_recipe_rating_data():
        """Charge les données de ratings."""
        global _recipe_rating_cache
        if _recipe_rating_cache is None:
            _recipe_rating_cache = pd.read_csv(cfg.data_rating_path)
        return _recipe_rating_cache


# Proxy pour compatibilité
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


# Variables pour compatibilité
recipe = _DataFrameProxy(get_recipe_data)
recipe_rating = _DataFrameProxy(get_recipe_rating_data)