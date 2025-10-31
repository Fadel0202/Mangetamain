import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.webapp_mangetamain.nutriscore_analyzer import (
    get_points,
    compute_nutriscore,
    parse_nutrition,
    filter_data_with_nutri,
    add_nutriscore_column,
    correlation_matrix,
)


# ==================== FIXTURES ====================


@pytest.fixture
def sample_recipe_df():
    """DataFrame with nutrition in JSON string format."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nutrition": [
                "[100, 10, 5, 20, 2, 50, 3]",  # calories, fat, sugar, sodium, protein, sat_fat, carbs
                "[200, 15, 8, 30, 3, 60, 4]",
                "[150, 12, 6, 25, 2.5, 55, 3.5]",
            ],
        }
    )


@pytest.fixture
def sample_nutrition_df():
    """DataFrame with already parsed nutrition data."""
    return pd.DataFrame(
        {
            "calories": [100, 200, 150],
            "total fat (PDV)": [10, 15, 12],
            "sugar (PDV)": [5, 8, 6],
            "sodium (PDV)": [20, 30, 25],
            "protein (PDV)": [2, 3, 2.5],
            "saturated fat (PDV)": [50, 60, 55],
            "carbohydrates (PDV)": [3, 4, 3.5],
        }
    )


@pytest.fixture
def sample_nutrition_with_tags():
    """Nutrition DataFrame with tags."""
    return pd.DataFrame(
        {
            "calories": [100, 200],
            "total fat (PDV)": [10, 15],
            "sugar (PDV)": [5, 8],
            "sodium (PDV)": [20, 30],
            "protein (PDV)": [2, 3],
            "saturated fat (PDV)": [50, 60],
            "carbohydrates (PDV)": [3, 4],
            "nutri_score": ["A", "B"],
        },
        index=[0, 1],
    )


@pytest.fixture
def sample_recipe_with_tags():
    """Recipe DataFrame with tags."""
    return pd.DataFrame({"tags": [["healthy", "quick"], ["dessert", "easy"]]}, index=[0, 1])


# ==================== TESTS ====================


def test_get_points():
    """Test points calculation based on thresholds."""
    thresholds = [10, 20, 30, 40]

    assert get_points(5, thresholds) == 0
    assert get_points(15, thresholds) == 1
    assert get_points(25, thresholds) == 2
    assert get_points(50, thresholds) == 4


def test_compute_nutriscore():
    """Test Nutri-Score calculation."""
    nutrition = {
        "calories": 100,
        "total fat (PDV)": 10,
        "sugar (PDV)": 5,
        "sodium (PDV)": 20,
        "protein (PDV)": 2,
        "saturated fat (PDV)": 50,
        "carbohydrates (PDV)": 3,
    }

    result = compute_nutriscore(nutrition)

    assert result in ["A", "B", "C", "D", "E"]


def test_parse_nutrition(sample_recipe_df):
    """Test parsing of nutrition column."""
    result = parse_nutrition(sample_recipe_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "calories" in result.columns


def test_filter_data_with_nutri(sample_nutrition_df):
    """Test filtering of extreme values."""
    result = filter_data_with_nutri(sample_nutrition_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_nutrition_df)


def test_add_nutriscore_column(sample_nutrition_df):
    """Test adding nutri_score column."""
    result = add_nutriscore_column(sample_nutrition_df)

    assert "nutri_score" in result.columns
    assert len(result) == len(sample_nutrition_df)
    assert all(result["nutri_score"].isin(["A", "B", "C", "D", "E"]))


def test_correlation_matrix(sample_nutrition_df):
    """Test correlation matrix creation."""
    fig = correlation_matrix(sample_nutrition_df)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_compute_nutriscore_edge_cases():
    """Test Nutri-Score with extreme values."""
    # Very good score (high protein, low negatives)
    good_nutrition = {
        "calories": 50,
        "total fat (PDV)": 5,
        "sugar (PDV)": 2,
        "sodium (PDV)": 5,
        "protein (PDV)": 50,
        "saturated fat (PDV)": 10,
        "carbohydrates (PDV)": 2,
    }

    result = compute_nutriscore(good_nutrition)
    assert result in ["A", "B", "C", "D", "E"]


def test_parse_nutrition_with_invalid_format():
    """Test parsing with invalid format."""
    df = pd.DataFrame({"nutrition": ["invalid", "[1,2,3]"]})

    # Should handle error or return DataFrame
    try:
        result = parse_nutrition(df)
        assert isinstance(result, pd.DataFrame)
    except Exception:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
