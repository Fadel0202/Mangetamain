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

recipe = pd.read_csv(cfg.data_path)
recipe_rating = pd.read_csv(cfg.data_rating_path)
