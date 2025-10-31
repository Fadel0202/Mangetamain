import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.webapp_mangetamain.recipe_complexity import (
    make_univariate_figs,
    make_pairplot_fig,
    make_corr_heatmap_fig,
)


# ==================== FIXTURE ====================


@pytest.fixture
def sample_df():
    """Test DataFrame."""
    return pd.DataFrame(
        {
            "minutes": [30, 45, 20, 60, 15],
            "n_ingredients": [5, 8, 4, 10, 3],
            "n_steps": [3, 6, 2, 8, 2],
        }
    )


# ==================== TESTS ====================


def test_make_univariate_figs_returns_figures(sample_df):
    """Check that make_univariate_figs returns 2 figures."""
    hist_fig, box_fig = make_univariate_figs(sample_df, "minutes")

    assert isinstance(hist_fig, plt.Figure)
    assert isinstance(box_fig, plt.Figure)

    plt.close("all")


def test_make_pairplot_fig_returns_figure(sample_df):
    """Check that make_pairplot_fig returns one figure."""
    features = ["minutes", "n_ingredients"]
    fig = make_pairplot_fig(sample_df, features)

    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_make_corr_heatmap_fig_returns_figure(sample_df):
    """Check that make_corr_heatmap_fig returns one figure."""
    features = ["minutes", "n_ingredients", "n_steps"]
    fig = make_corr_heatmap_fig(sample_df, features)

    assert isinstance(fig, plt.Figure)
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
