import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.webapp_mangetamain.tag_analyzer import (
    parse_tags,
    get_all_tags_of_interest,
    create_tag_category_mapping,
    get_general_tags_statistics,
    analyze_tags_distribution,
    create_tag_recipes_dataset,
    filter_tags_of_interest,
    get_summary_statistics,
    find_best_tags,
)

# ==================== FIXTURES ====================


@pytest.fixture
def sample_recipes_df():
    """Test DataFrame with recipe data."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "tags": ["['italian', 'easy']", "['mexican']", "['desserts']"],
            "minutes": [30, 15, 45],
            "n_ingredients": [8, 5, 10],
            "n_steps": [5, 3, 7],
        }
    )


@pytest.fixture
def sample_tag_stats():
    """Tag statistics with categories."""
    return pd.DataFrame(
        {
            "tag": ["italian", "mexican", "desserts"],
            "category": ["Cuisine", "Cuisine", "Dish Type"],
            "n_recipes": [100, 80, 150],
            "avg_minutes": [35.5, 25.0, 40.0],
            "avg_ingredients": [8.2, 6.5, 9.0],
            "avg_steps": [6.0, 4.5, 7.0],
        }
    )


# ==================== TESTS ====================


def test_parse_tags(sample_recipes_df):
    """Test tag parsing."""
    result = parse_tags(sample_recipes_df["tags"])
    assert len(result) == 3
    assert result.iloc[0] == ["italian", "easy"]


def test_get_all_tags_of_interest():
    """Test retrieval of tags list."""
    result = get_all_tags_of_interest()
    assert isinstance(result, list)
    assert "italian" in result


def test_create_tag_category_mapping():
    """Test tag->category mapping creation."""
    mapping = create_tag_category_mapping()
    assert mapping["italian"] == "Cuisine"


def test_get_general_tags_statistics(sample_recipes_df):
    """Test general statistics."""
    stats = get_general_tags_statistics(sample_recipes_df)
    assert stats["total_recipes"] == 3


def test_analyze_tags_distribution(sample_recipes_df):
    """Test distribution analysis."""
    result = analyze_tags_distribution(sample_recipes_df)
    assert isinstance(result, pd.DataFrame)
    assert "tag" in result.columns


def test_create_tag_recipes_dataset(sample_recipes_df):
    """Test tag-recipe dataset creation."""
    tag_stats, _ = create_tag_recipes_dataset(sample_recipes_df, min_recipes_per_tag=1)
    assert isinstance(tag_stats, pd.DataFrame)
    assert "n_recipes" in tag_stats.columns


def test_filter_tags_of_interest():
    """Test filtering of tags of interest."""
    df = pd.DataFrame(
        {
            "tag": ["italian", "unknown"],
            "n_recipes": [100, 50],
            "avg_minutes": [30, 40],
            "avg_ingredients": [8, 10],
            "avg_steps": [5, 7],
        }
    )
    result = filter_tags_of_interest(df)
    assert "category" in result.columns


def test_get_summary_statistics(sample_tag_stats):
    """Test summary by category."""
    summary = get_summary_statistics(sample_tag_stats)
    assert isinstance(summary, pd.DataFrame)


def test_find_best_tags(sample_tag_stats):
    """Test best tags search."""
    result = find_best_tags(sample_tag_stats, top_n=2)
    assert "Fastest" in result
    assert "Most popular" in result


def test_empty_dataframe():
    """Test with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["id", "tags", "minutes", "n_ingredients", "n_steps"])
    result = get_general_tags_statistics(empty_df)
    assert result["total_recipes"] == 0


def test_single_recipe():
    """Test with single recipe."""
    df = pd.DataFrame(
        {"id": [1], "tags": ["['italian']"], "minutes": [30], "n_ingredients": [5], "n_steps": [3]}
    )
    stats = get_general_tags_statistics(df)
    assert stats["total_recipes"] == 1


def test_complete_workflow(sample_recipes_df):
    """Test complete workflow."""
    stats = get_general_tags_statistics(sample_recipes_df)
    tag_stats, _ = create_tag_recipes_dataset(sample_recipes_df, min_recipes_per_tag=1)

    assert stats is not None
    assert len(tag_stats) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
