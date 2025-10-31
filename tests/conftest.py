"""Configuration pytest et fixtures pour les tests."""
import pytest
import pandas as pd
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_recipe_data():
    """
    Mock automatique des données de recettes pour tous les tests.
    
    Cette fixture est appliquée automatiquement à tous les tests,
    évitant ainsi le besoin de fichiers CSV en CI/CD.
    """
    # Créer un DataFrame de test pour recipe
    mock_recipe_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': [
            'Test Recipe 1',
            'Test Recipe 2',
            'Test Recipe 3',
            'Healthy Recipe',
            'Quick Meal'
        ],
        'minutes': [30, 45, 60, 20, 15],
        'contributor_id': [100, 101, 102, 103, 104],
        'submitted': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
        'tags': [
            ['tag1', 'tag2'],
            ['tag3', 'tag4'],
            ['tag5', 'tag6'],
            ['healthy', 'diet'],
            ['quick', 'easy']
        ],
        'nutrition': [
            [100, 20, 30, 5, 10, 15, 20],
            [200, 30, 40, 10, 15, 20, 25],
            [150, 25, 35, 7, 12, 18, 22],
            [80, 15, 25, 3, 8, 10, 15],
            [120, 22, 28, 6, 11, 16, 19]
        ],
        'n_steps': [5, 7, 10, 3, 4],
        'steps': [
            ['step1', 'step2', 'step3', 'step4', 'step5'],
            ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7'],
            ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8', 'step9', 'step10'],
            ['step1', 'step2', 'step3'],
            ['step1', 'step2', 'step3', 'step4']
        ],
        'description': [
            'Description 1',
            'Description 2',
            'Description 3',
            'Healthy description',
            'Quick description'
        ],
        'ingredients': [
            ['ingredient1', 'ingredient2', 'ingredient3'],
            ['ingredient4', 'ingredient5', 'ingredient6'],
            ['ingredient7', 'ingredient8', 'ingredient9'],
            ['ingredient10', 'ingredient11'],
            ['ingredient12', 'ingredient13', 'ingredient14']
        ],
        'n_ingredients': [8, 10, 12, 5, 6],
    })
    
    # Créer un DataFrame de test pour recipe_rating
    mock_rating_df = pd.DataFrame({
        'recipe_id': [1, 2, 3, 4, 5],
        'user_id': [1000, 1001, 1002, 1003, 1004],
        'rating': [4.5, 5.0, 3.5, 4.0, 4.8],
        'review': ['Good', 'Excellent', 'Average', 'Nice', 'Great'],
    })
    
    # Mock les fonctions get_recipe_data() et get_recipe_rating_data()
    with patch('webapp_mangetamain.load_config.get_recipe_data', return_value=mock_recipe_df):
        with patch('webapp_mangetamain.load_config.get_recipe_rating_data', return_value=mock_rating_df):
            # Mock aussi pandas.read_csv au cas où
            with patch('pandas.read_csv') as mock_read_csv:
                # Configure le mock pour retourner le bon DataFrame selon le fichier
                def side_effect(filepath, *args, **kwargs):
                    if 'rating' in str(filepath).lower():
                        return mock_rating_df
                    return mock_recipe_df
                
                mock_read_csv.side_effect = side_effect
                yield {
                    'recipe': mock_recipe_df,
                    'recipe_rating': mock_rating_df
                }
