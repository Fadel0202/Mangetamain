"""
Tags Analyzer Module
Analysis of tags and their relationships with recipe metrics
"""

import ast
from typing import Dict, List, Tuple
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

TAGS_OF_INTEREST = {
    'Cuisine': ['italian', 'mexican', 'asian', 'french', 'chinese', 'greek',
                'indian', 'thai', 'japanese', 'spanish', 'middle-eastern'],

    'Dish Type': ['desserts', 'main-dish', 'breakfast', 'appetizers',
                  'side-dishes', 'lunch', 'brunch', 'salads'],

    'Characteristics': ['easy', 'healthy', 'low-fat', 'low-calorie',
                        'low-carb', 'vegetarian', 'vegan', 'beginner-cook'],

    'Time': ['15-minutes-or-less', '30-minutes-or-less', '60-minutes-or-less',
             '4-hours-or-less'],

    'Beverages': ['cocktails', 'smoothies', 'beverages', 'shakes',
                  'non-alcoholic', 'punch']
}


def parse_tags(tags_series: pd.Series) -> pd.Series:
    """
    Parse tags stored as strings into lists.

    Args:
        tags_series: Series containing tags as string format

    Returns:
        Series with tags parsed as lists
    """
    def safe_parse(x):
        try:
            if pd.isna(x):
                return []
            return ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return []

    return tags_series.apply(safe_parse)


def get_all_tags_of_interest() -> List[str]:
    """
    Return the list of all tags of interest.

    Returns:
        List of all tags of interest
    """
    return [tag for category in TAGS_OF_INTEREST.values() for tag in category]


def create_tag_category_mapping() -> Dict[str, str]:
    """
    Create a dictionary mapping tag -> category.

    Returns:
        Dictionary {tag: category}
    """
    category_map = {}
    for category, tags in TAGS_OF_INTEREST.items():
        for tag in tags:
            category_map[tag] = category
    return category_map


def get_general_tags_statistics(recipes_df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate general statistics on tags.

    Args:
        recipes_df: DataFrame containing a 'tags' column

    Returns:
        Dictionary containing general statistics
    """
    # Parse tags
    recipes_df['tags_parsed'] = parse_tags(recipes_df['tags'])

    # Statistics on number of tags per recipe
    tags_per_recipe = recipes_df['tags_parsed'].apply(len)

    # Create a list of all tags
    all_tags = []
    for tags_list in recipes_df['tags_parsed']:
        all_tags.extend(tags_list)

    # Count occurrences
    tag_counts = pd.Series(all_tags).value_counts()
    avg_tags_general = len(all_tags) / len(tag_counts) if len(tag_counts) > 0 else 0

    stats = {
        'total_recipes': len(recipes_df),
        'tags_per_recipe_stats': tags_per_recipe.describe(),
        'tags_per_recipe_mean': tags_per_recipe.mean(),
        'tags_per_recipe_median': tags_per_recipe.median(),
        'tags_per_recipe_min': tags_per_recipe.min(),
        'tags_per_recipe_max': tags_per_recipe.max(),
        'total_unique_tags': len(tag_counts),
        'total_tags': len(all_tags),
        'avg_tags_general': avg_tags_general,
        'tag_counts_stats': tag_counts.describe(),
        'top_20_tags': tag_counts.head(20)
    }

    return stats

def analyze_tags_distribution(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze tag distribution in the dataset.

    Args:
        recipes_df: DataFrame containing a 'tags' column

    Returns:
        DataFrame with statistics per tag
    """
    # Parse tags
    if 'tags_parsed' not in recipes_df.columns:
        recipes_df['tags_parsed'] = parse_tags(recipes_df['tags'])

    # Create a list of all tags
    all_tags = []
    for tags_list in recipes_df['tags_parsed']:
        all_tags.extend(tags_list)

    # Count occurrences
    tag_counts = pd.Series(all_tags).value_counts().reset_index()
    tag_counts.columns = ['tag', 'count']

    return tag_counts


def create_tag_recipes_dataset(
    recipes_df: pd.DataFrame,
    min_recipes_per_tag: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a dataset for tag analysis with aggregated metrics.

    Args:
        recipes_df: DataFrame of recipes with necessary columns
        min_recipes_per_tag: Minimum number of recipes to include a tag

    Returns:
        Tuple (tag_stats, tag_recipes_df)
    """
    # Parse tags
    if 'tags_parsed' not in recipes_df.columns:
        recipes_df['tags_parsed'] = parse_tags(recipes_df['tags'])

    # Create one row per tag-recipe pair
    tag_recipe_pairs = []
    for idx, row in recipes_df.iterrows():
        for tag in row['tags_parsed']:
            tag_recipe_pairs.append({
                'tag': tag,
                'recipe_id': row.get('id', idx),
                'minutes': row.get('minutes', 0),
                'n_ingredients': row.get('n_ingredients', 0),
                'n_steps': row.get('n_steps', 0)
            })

    tag_recipes_df = pd.DataFrame(tag_recipe_pairs)

    # Statistics per tag
    tag_stats = tag_recipes_df.groupby('tag').agg({
        'recipe_id': 'count',
        'minutes': 'mean',
        'n_ingredients': 'mean',
        'n_steps': 'mean'
    }).reset_index()

    tag_stats.columns = ['tag', 'n_recipes', 'avg_minutes',
                         'avg_ingredients', 'avg_steps']

    # Filter tags with enough recipes
    tag_stats = tag_stats[tag_stats['n_recipes'] >= min_recipes_per_tag].copy()
    tag_stats = tag_stats.sort_values('n_recipes', ascending=False)

    return tag_stats, tag_recipes_df


def filter_tags_of_interest(tag_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only tags of interest.

    Args:
        tag_stats: DataFrame with statistics per tag

    Returns:
        Filtered DataFrame with 'category' column added
    """
    all_tags = get_all_tags_of_interest()
    df_filtered = tag_stats[tag_stats['tag'].isin(all_tags)].copy()

    # Add category
    category_map = create_tag_category_mapping()
    df_filtered['category'] = df_filtered['tag'].map(category_map)

    return df_filtered


def plot_top_tags_by_metric(
    tag_stats: pd.DataFrame,
    metric: str = 'n_recipes',
    top_n: int = 20,
    title: str = None
) -> plt.Figure:
    """
    Create a chart of top N tags by a metric.

    Args:
        tag_stats: DataFrame with statistics per tag
        metric: Metric to use for ranking
        top_n: Number of tags to display
        title: Chart title

    Returns:
        Matplotlib Figure
    """
    df_top = tag_stats.nlargest(top_n, metric)

    metric_labels = {
        'n_recipes': 'Number of recipes',
        'avg_minutes': 'Average time (min)',
        'avg_ingredients': 'Average number of ingredients',
        'avg_steps': 'Average number of steps'
    }

    if title is None:
        title = f"Top {top_n} tags by {metric_labels.get(metric, metric)}"

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

    # Create horizontal bar chart
    colors = plt.cm.viridis(df_top[metric].values / df_top[metric].values.max())
    ax.barh(range(len(df_top)), df_top[metric].values, color=colors)

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['tag'].values)
    ax.set_xlabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    return fig

def create_heatmap_tags_metrics(tag_stats_filtered: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap of tags vs metrics.

    Args:
        tag_stats_filtered: DataFrame with filtered tags

    Returns:
        Matplotlib Figure
    """
    # Select metrics and normalize
    metrics = ['avg_minutes', 'avg_ingredients', 'avg_steps']
    df_norm = tag_stats_filtered.copy()

    for metric in metrics:
        min_val = df_norm[metric].min()
        max_val = df_norm[metric].max()
        df_norm[f'{metric}_norm'] = (df_norm[metric] - min_val) / (max_val - min_val)

    # Prepare data for heatmap
    metric_labels = ['Time', 'Ingredients', 'Steps']
    z_data = df_norm[['avg_minutes_norm', 'avg_ingredients_norm', 'avg_steps_norm']].T.values

    fig, ax = plt.subplots(figsize=(max(12, len(df_norm) * 0.3), 5))

    im = ax.imshow(z_data, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(df_norm)))
    ax.set_xticklabels(df_norm['tag'].values, rotation=90, ha='right')
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized value', rotation=270, labelpad=20)

    ax.set_title('Heatmap: Tags vs Metrics (normalized values)', fontsize=14, pad=15)
    ax.set_xlabel('Tags', fontsize=12)
    ax.set_ylabel('Metrics', fontsize=12)

    fig.tight_layout()
    return fig


def get_summary_statistics(tag_stats_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics by category.

    Args:
        tag_stats_filtered: DataFrame with filtered tags

    Returns:
        DataFrame with statistics by category
    """
    summary = tag_stats_filtered.groupby('category').agg({
        'tag': 'count',
        'n_recipes': 'sum',
        'avg_minutes': 'mean',
        'avg_ingredients': 'mean',
        'avg_steps': 'mean'
    }).round(2)

    summary.columns = [
        'Number of tags',
        'Total recipes',
        'Average time',
        'Average ingredients',
        'Average steps'
    ]

    return summary


def find_best_tags(
    tag_stats_filtered: pd.DataFrame,
    top_n: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Identify the best tags according to different criteria.

    Args:
        tag_stats_filtered: DataFrame with filtered tags
        top_n: Number of tags to return per criterion

    Returns:
        Dictionary {criterion: DataFrame}
    """
    results = {
        'Fastest': tag_stats_filtered.nsmallest(
            top_n, 'avg_minutes'
        )[['tag', 'category', 'avg_minutes', 'n_recipes']],

        'Simplest': tag_stats_filtered.nsmallest(
            top_n, 'avg_ingredients'
        )[['tag', 'category', 'avg_ingredients', 'n_recipes']],

        'Most popular': tag_stats_filtered.nlargest(
            top_n, 'n_recipes'
        )[['tag', 'category', 'n_recipes', 'avg_minutes']],

        'Fewest steps': tag_stats_filtered.nsmallest(
            top_n, 'avg_steps'
        )[['tag', 'category', 'avg_steps', 'n_recipes']]
    }

    return results

def plot_tags_per_recipe_distribution(recipes_df: pd.DataFrame) -> None:
    """
    Create and display a histogram of the distribution of tags per recipe.

    Args:
        recipes_df: DataFrame containing a 'tags_parsed' column
    """
    tags_per_recipe = recipes_df['tags_parsed'].apply(len)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    ax.hist(tags_per_recipe, bins=50, color='#636EFA', alpha=0.7, edgecolor='black')

    # Add mean and median lines
    mean_val = tags_per_recipe.mean()
    median_val = tags_per_recipe.median()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.1f}')

    ax.set_xlabel('Number of tags per recipe', fontsize=12)
    ax.set_ylabel('Number of recipes', fontsize=12)
    ax.set_title('Distribution of number of tags per recipe', fontsize=14, pad=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_top_tags(tag_counts: pd.Series, top_n: int = 20) -> None:
    """
    Create and display a chart of the most frequent tags.

    Args:
        tag_counts: Series with tag counts
        top_n: Number of tags to display
    """
    top_tags = tag_counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

    # Create horizontal bar chart
    colors = plt.cm.viridis(top_tags.values / top_tags.values.max())
    ax.barh(range(len(top_tags)), top_tags.values, color=colors)

    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags.index)
    ax.set_xlabel('Number of occurrences', fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    ax.set_title(f'Top {top_n} most frequent tags', fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_tag_frequency_distribution(tag_counts: pd.Series) -> None:
    """
    Create and display a histogram of tag frequency distribution.

    Args:
        tag_counts: Series with tag counts
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with log scale on y-axis
    ax.hist(tag_counts.values, bins=50, color='#EF553B', alpha=0.7, edgecolor='black')
    ax.set_yscale('log')

    ax.set_xlabel('Number of occurrences per tag', fontsize=12)
    ax.set_ylabel('Number of tags (log scale)', fontsize=12)
    ax.set_title('Distribution of tag frequency', fontsize=14, pad=15)
    ax.grid(axis='both', alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_categories_comparison(tag_stats_filtered: pd.DataFrame) -> None:
    """
    Compare and display tag categories across multiple metrics.

    Args:
        tag_stats_filtered: DataFrame with filtered tags and 'category' column
    """
    category_stats = tag_stats_filtered.groupby('category').agg({
        'avg_minutes': 'mean',
        'avg_ingredients': 'mean',
        'avg_steps': 'mean',
        'n_recipes': 'sum'
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('avg_minutes', 'Average time (min)', axes[0, 0]),
        ('avg_ingredients', 'Average ingredients', axes[0, 1]),
        ('avg_steps', 'Average steps', axes[1, 0]),
        ('n_recipes', 'Total recipes', axes[1, 1])
    ]

    for metric, label, ax in metrics:
        ax.bar(category_stats['category'], category_stats[metric],
               color='lightblue', edgecolor='navy', alpha=0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel('Category', fontsize=11)
        ax.set_title(label, fontsize=12, pad=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Comparison of tag categories', fontsize=16, y=0.995)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_category_detail(
    tag_stats_filtered: pd.DataFrame,
    category: str,
    metric: str = 'avg_minutes',
    top_n: int = 10
) -> None:
    """
    Create and display a detailed chart for a specific category.

    Args:
        tag_stats_filtered: DataFrame with filtered tags
        category: Category to display
        metric: Metric to display
        top_n: Number of tags to display
    """
    df_cat = tag_stats_filtered[
        tag_stats_filtered['category'] == category
    ].copy()

    # Sort by metric
    if metric == 'avg_minutes':
        df_cat = df_cat.nsmallest(top_n, metric)
    else:
        df_cat = df_cat.nlargest(top_n, metric)

    metric_labels = {
        'avg_minutes': 'Average time (min)',
        'avg_ingredients': 'Average number of ingredients',
        'avg_steps': 'Average number of steps',
        'n_recipes': 'Number of recipes'
    }

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    # Create horizontal bar chart
    colors = plt.cm.viridis(df_cat[metric].values / df_cat[metric].values.max())
    ax.barh(range(len(df_cat)), df_cat[metric].values, color=colors)

    ax.set_yticks(range(len(df_cat)))
    ax.set_yticklabels(df_cat['tag'].values)
    ax.set_xlabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    ax.set_title(f"{category} - Top {top_n} by {metric_labels.get(metric, metric)}",
                 fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


