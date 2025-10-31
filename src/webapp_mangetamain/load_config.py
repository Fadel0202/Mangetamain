"""Load config from config.json"""

import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests


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
GITHUB_RELEASE_URL = "https://github.com/Fadel0202/Mangetamain/releases/download/v1.0.0/data_with_artifacts.zip"


def download_data_if_needed():
    """Télécharge les données depuis GitHub Release si elles n'existent pas."""
    data_dir = Path("data")
    artifacts_dir = Path("artifacts")
    recipe_file = data_dir / "RAW_recipes.csv"
    co_occurrence_file = artifacts_dir / "co_occurrence.csv"

    # Si les données ET artifacts existent déjà, ne rien faire
    if recipe_file.exists() and co_occurrence_file.exists():
        print("Données et artifacts déjà présents")
        return

    print("Téléchargement des données depuis GitHub Release (176 Mo, ~30-40 secondes)...")

    try:
        # Créer les dossiers
        data_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)

        # Télécharger le zip
        response = requests.get(GITHUB_RELEASE_URL, timeout=300, stream=True)
        response.raise_for_status()

        zip_path = "data_with_artifacts.zip"
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Afficher la progression tous les 10 Mo
                    if downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100) if total_size else 0
                        print(f"   Progression: {progress:.1f}%")

        print("Extraction des données...")
        # Dézipper
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        # Nettoyer
        os.remove(zip_path)
        print("Données téléchargées et extraites avec succès!")

    except Exception as e:
        print(f"Erreur lors du téléchargement des données: {e}")
        raise


# Télécharger les données au chargement du module (une seule fois)
download_data_if_needed()

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