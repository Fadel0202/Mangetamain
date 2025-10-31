"""
Tags Analyzer Module
Analysis of tags and their relationships with recipe metrics
"""

import ast
from typing import Dict, List, Tuple, Any, Iterable

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

TAGS_OF_INTEREST = {
    "Cuisine": [
        "italian",
        "mexican",
        "asian",
        "french",
        "chinese",
        "greek",
        "indian",
        "thai",
        "japanese",
        "spanish",
        "middle-eastern",
    ],
    "Dish Type": [
        "desserts",
        "main-dish",
        "breakfast",
        "appetizers",
        "side-dishes",
        "lunch",
        "brunch",
        "salads",
    ],
    "Characteristics": [
        "easy",
        "healthy",
        "low-fat",
        "low-calorie",
        "low-carb",
        "vegetarian",
        "vegan",
        "beginner-cook",
    ],
    "Time": [
        "15-minutes-or-less",
        "30-minutes-or-less",
        "60-minutes-or-less",
        "4-hours-or-less",
    ],
    "Beverages": [
        "cocktails",
        "smoothies",
        "beverages",
        "shakes",
        "non-alcoholic",
        "punch",
    ],
}


def parse_tags(tags_series: pd.Series) -> pd.Series:
    """
    Parse tags stored as strings into lists of lowercase strings.

    Args:
        tags_series: Series containing tags as string format

    Returns:
        Series with tags parsed as lists of lowercase strings
    """
    def safe_parse(x):
        try:
            if pd.isna(x):
                return []
            # Parse string representation of list; ensure result is a list
            value = ast.literal_eval(x) if isinstance(x, str) else x
            if not isinstance(value, list):
                value = [value]
            # Normalize each tag: string, lowercase, stripped
            return [str(tag).lower().strip() for tag in value]
        except Exception:
            return []

    return tags_series.apply(safe_parse)


def get_all_tags_of_interest() -> List[str]:
    """Return the list of all tags of interest (already in lowercase)."""
    return [tag for category in TAGS_OF_INTEREST.values() for tag in category]


def create_tag_category_mapping() -> Dict[str, str]:
    """Create a dictionary mapping tag -> category."""
    category_map: Dict[str, str] = {}
    for category, tags in TAGS_OF_INTEREST.items():
        for tag in tags:
            category_map[tag] = category
    return category_map


def get_general_tags_statistics(recipes_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate general statistics on tags.

    Args:
        recipes_df: DataFrame (or proxy) containing a 'tags' column

    Returns:
        Dictionary of general statistics
    """
    if "tags" not in recipes_df.columns:
        raise KeyError("La colonne 'tags' est absente du DataFrame fourni.")

    tags_parsed = parse_tags(recipes_df["tags"])
    tags_per_recipe = tags_parsed.str.len()
    tags_exploded = tags_parsed.explode()
    tag_counts = tags_exploded.value_counts()
    avg_tags_general = (
        tags_per_recipe.sum() / len(tag_counts) if len(tag_counts) > 0 else 0
    )

    return {
        "total_recipes": len(recipes_df),
        "tags_per_recipe_stats": tags_per_recipe.describe(),
        "tags_per_recipe_mean": tags_per_recipe.mean(),
        "tags_per_recipe_median": tags_per_recipe.median(),
        "tags_per_recipe_min": tags_per_recipe.min(),
        "tags_per_recipe_max": tags_per_recipe.max(),
        "total_unique_tags": len(tag_counts),
        "total_tags": tags_per_recipe.sum(),
        "avg_tags_general": avg_tags_general,
        "tag_counts_stats": tag_counts.describe(),
        "top_20_tags": tag_counts.head(20),
    }


def analyze_tags_distribution(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze tag distribution in the dataset.

    Args:
        recipes_df: DataFrame containing a 'tags' column

    Returns:
        DataFrame with two columns: tag and count
    """
    if "tags_parsed" not in recipes_df.columns:
        recipes_df["tags_parsed"] = parse_tags(recipes_df["tags"])

    tag_counts = (
        recipes_df["tags_parsed"]
        .explode()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "tag", "tags_parsed": "count"})
    )
    return tag_counts


def create_tag_recipes_dataset(
    recipes_df: pd.DataFrame, min_recipes_per_tag: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a dataset for tag analysis with aggregated metrics.

    Args:
        recipes_df: DataFrame of recipes with necessary columns
        min_recipes_per_tag: Minimum number of recipes to include a tag

    Returns:
        Tuple (tag_stats, tag_recipes_df)
    """
    # Ensure tags_parsed column exists (lowercase tags)
    if "tags_parsed" not in recipes_df.columns:
        recipes_df["tags_parsed"] = parse_tags(recipes_df["tags"])

    # Explode tags and select useful columns
    cols = ["id", "tags_parsed", "minutes", "n_ingredients", "n_steps"]
    df_exp = (
        recipes_df[cols]
        .explode("tags_parsed")
        .dropna(subset=["tags_parsed"])
        .rename(columns={"tags_parsed": "tag", "id": "recipe_id"})
    )
    # Normaliser le tag (double sécurité)
    df_exp["tag"] = df_exp["tag"].astype(str).str.lower().str.strip()

    tag_stats = (
        df_exp.groupby("tag")
        .agg(
            n_recipes=("recipe_id", "count"),
            avg_minutes=("minutes", "mean"),
            avg_ingredients=("n_ingredients", "mean"),
            avg_steps=("n_steps", "mean"),
        )
        .reset_index()
    )

    # Filter tags with enough recipes
    tag_stats = (
        tag_stats[tag_stats["n_recipes"] >= min_recipes_per_tag]
        .sort_values("n_recipes", ascending=False)
        .reset_index(drop=True)
    )

    return tag_stats, df_exp


def filter_tags_of_interest(tag_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only tags of interest and add category column.

    Args:
        tag_stats: DataFrame with statistics per tag

    Returns:
        Filtered DataFrame with a 'category' column
    """
    all_tags = get_all_tags_of_interest()
    df_filtered = tag_stats[tag_stats["tag"].isin(all_tags)].copy()
    category_map = create_tag_category_mapping()
    df_filtered["category"] = df_filtered["tag"].map(category_map)
    return df_filtered


def plot_top_tags_by_metric(
    tag_stats: pd.DataFrame,
    metric: str = "n_recipes",
    top_n: int = 20,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a chart of top N tags by a metric.

    Returns:
        Matplotlib Figure
    """
    df_top = tag_stats.nlargest(top_n, metric)
    metric_labels = {
        "n_recipes": "Number of recipes",
        "avg_minutes": "Average time (min)",
        "avg_ingredients": "Average number of ingredients",
        "avg_steps": "Average number of steps",
    }

    if title is None:
        title = f"Top {top_n} tags by {metric_labels.get(metric, metric)}"

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    colors = plt.cm.viridis(df_top[metric].values / df_top[metric].values.max())
    ax.barh(range(len(df_top)), df_top[metric].values, color=colors)

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["tag"].values)
    ax.set_xlabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_ylabel("Tag", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


def create_heatmap_tags_metrics(tag_stats_filtered: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap of tags vs metrics.

    Returns:
        Matplotlib Figure
    """
    metrics = ["avg_minutes", "avg_ingredients", "avg_steps"]
    df_norm = tag_stats_filtered.copy()
    for metric in metrics:
        min_val = df_norm[metric].min()
        max_val = df_norm[metric].max()
        df_norm[f"{metric}_norm"] = (df_norm[metric] - min_val) / (max_val - min_val)

    metric_labels = ["Time", "Ingredients", "Steps"]
    z_data = df_norm[
        ["avg_minutes_norm", "avg_ingredients_norm", "avg_steps_norm"]
    ].T.values

    fig, ax = plt.subplots(figsize=(max(12, len(df_norm) * 0.3), 5))
    im = ax.imshow(z_data, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(df_norm)))
    ax.set_xticklabels(df_norm["tag"].values, rotation=90, ha="right")
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized value", rotation=270, labelpad=20)

    ax.set_title("Heatmap: Tags vs Metrics (normalized values)", fontsize=14, pad=15)
    ax.set_xlabel("Tags", fontsize=12)
    ax.set_ylabel("Metrics", fontsize=12)

    fig.tight_layout()
    return fig


def get_summary_statistics(tag_stats_filtered: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics by category."""
    summary = (
        tag_stats_filtered.groupby("category")
        .agg(
            tag=("tag", "count"),
            n_recipes=("n_recipes", "sum"),
            avg_minutes=("avg_minutes", "mean"),
            avg_ingredients=("avg_ingredients", "mean"),
            avg_steps=("avg_steps", "mean"),
        )
        .round(2)
    )
    summary.columns = [
        "Number of tags",
        "Total recipes",
        "Average time",
        "Average ingredients",
        "Average steps",
    ]
    return summary.reset_index()


def find_best_tags(
    tag_stats_filtered: pd.DataFrame, top_n: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Identify the best tags according to different criteria.

    Returns:
        Dictionary {criterion: DataFrame}
    """
    return {
        "Fastest": tag_stats_filtered.nsmallest(top_n, "avg_minutes")[
            ["tag", "category", "avg_minutes", "n_recipes"]
        ],
        "Simplest": tag_stats_filtered.nsmallest(top_n, "avg_ingredients")[
            ["tag", "category", "avg_ingredients", "n_recipes"]
        ],
        "Most popular": tag_stats_filtered.nlargest(top_n, "n_recipes")[
            ["tag", "category", "n_recipes", "avg_minutes"]
        ],
        "Fewest steps": tag_stats_filtered.nsmallest(top_n, "avg_steps")[
            ["tag", "category", "avg_steps", "n_recipes"]
        ],
    }


def plot_tags_per_recipe_distribution(recipes_df: pd.DataFrame) -> None:
    """Create and display a histogram of the distribution of tags per recipe."""
    tags_per_recipe = recipes_df["tags_parsed"].str.len()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tags_per_recipe, bins=50, color="#636EFA", alpha=0.7, edgecolor="black")

    mean_val = tags_per_recipe.mean()
    median_val = tags_per_recipe.median()
    ax.axvline(
        mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}"
    )
    ax.axvline(
        median_val,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.1f}",
    )

    ax.set_xlabel("Number of tags per recipe", fontsize=12)
    ax.set_ylabel("Number of recipes", fontsize=12)
    ax.set_title("Distribution of number of tags per recipe", fontsize=14, pad=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_top_tags(tag_counts: pd.Series, top_n: int = 20) -> None:
    """Create and display a chart of the most frequent tags."""
    top_tags = tag_counts.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    colors = plt.cm.viridis(top_tags.values / top_tags.values.max())
    ax.barh(range(len(top_tags)), top_tags.values, color=colors)

    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags.index)
    ax.set_xlabel("Number of occurrences", fontsize=12)
    ax.set_ylabel("Tag", fontsize=12)
    ax.set_title(f"Top {top_n} most frequent tags", fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_tag_frequency_distribution(tag_counts: pd.Series) -> None:
    """Create and display a histogram of tag frequency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tag_counts.values, bins=50, color="#EF553B", alpha=0.7, edgecolor="black")
    ax.set_yscale("log")

    ax.set_xlabel("Number of occurrences per tag", fontsize=12)
    ax.set_ylabel("Number of tags (log scale)", fontsize=12)
    ax.set_title("Distribution of tag frequency", fontsize=14, pad=15)
    ax.grid(axis="both", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_categories_comparison(tag_stats_filtered: pd.DataFrame) -> None:
    """Compare and display tag categories across multiple metrics."""
    category_stats = (
        tag_stats_filtered.groupby("category")
        .agg(
            avg_minutes=("avg_minutes", "mean"),
            avg_ingredients=("avg_ingredients", "mean"),
            avg_steps=("avg_steps", "mean"),
            n_recipes=("n_recipes", "sum"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("avg_minutes", "Average time (min)", axes[0, 0]),
        ("avg_ingredients", "Average ingredients", axes[0, 1]),
        ("avg_steps", "Average steps", axes[1, 0]),
        ("n_recipes", "Total recipes", axes[1, 1]),
    ]

    for metric, label, ax in metrics:
        ax.bar(
            category_stats["category"],
            category_stats[metric],
            color="lightblue",
            edgecolor="navy",
            alpha=0.7,
        )
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel("Category", fontsize=11)
        ax.set_title(label, fontsize=12, pad=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparison of tag categories", fontsize=16, y=0.995)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_category_detail(
    tag_stats_filtered: pd.DataFrame,
    category: str,
    metric: str = "avg_minutes",
    top_n: int = 10,
) -> None:
    """Create and display a detailed chart for a specific category."""
    df_cat = tag_stats_filtered[tag_stats_filtered["category"] == category].copy()
    df_cat = (
        df_cat.nsmallest(top_n, metric)
        if metric == "avg_minutes"
        else df_cat.nlargest(top_n, metric)
    )

    metric_labels = {
        "avg_minutes": "Average time (min)",
        "avg_ingredients": "Average number of ingredients",
        "avg_steps": "Average number of steps",
        "n_recipes": "Number of recipes",
    }

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.viridis(df_cat[metric].values / df_cat[metric].values.max())
    ax.barh(range(len(df_cat)), df_cat[metric].values, color=colors)

    ax.set_yticks(range(len(df_cat)))
    ax.set_yticklabels(df_cat["tag"].values)
    ax.set_xlabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_ylabel("Tag", fontsize=12)
    ax.set_title(
        f"{category} - Top {top_n} by {metric_labels.get(metric, metric)}",
        fontsize=14,
        pad=15,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
