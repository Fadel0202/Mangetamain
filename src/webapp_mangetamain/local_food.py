import utils.filter_data as filter_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
def plot_cuisine_distributions(recipes_with_continent: pd.DataFrame):
    """
    Displays three boxplots comparing recipe characteristics across continents:
    - log_minutes (log-transformed preparation time)
    - n_steps (number of steps)
    - n_ingredients (number of ingredients)

    Parameters:
        recipes_with_continent : DataFrame containing
            ['continent', 'log_minutes', 'n_steps', 'n_ingredients']

    Returns:
        fig : matplotlib Figure
    """
    cols = ["log_minutes", "n_steps", "n_ingredients"]
    titles = {
        "log_minutes": "Preparation time (log scale)",
        "n_steps": "Number of steps",
        "n_ingredients": "Number of ingredients"
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    for i, col in enumerate(cols):
        sns.boxplot(
            x="continent", y=col, data=recipes_with_continent,
            ax=axes[i], palette="viridis"
        )
        axes[i].set_title(f"Distribution of {titles[col]} by continent")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    fig.tight_layout()
    return fig


def plot_top_ingredients_by_continent(recipes_with_continent: pd.DataFrame, top_n: int = 10, global_threshold: float = 0.30):
    """
    Displays the most commonly used ingredients for each continent,
    while excluding globally ubiquitous ingredients.

    Parameters:
        recipes_with_continent : DataFrame with ['id', 'continent', 'ingredients']
        top_n : number of ingredients to display per continent
        global_threshold : exclude ingredients that appear in more than X% of all recipes
    """
    df = recipes_with_continent.copy()
    df["ingredients"] = df["ingredients"].str.lower().str.strip()

    # --- 1. Compute global frequency of each ingredient
    total_recipes = df["id"].nunique()
    global_freq = (
        df.groupby("ingredients")["id"]
        .nunique()
        .div(total_recipes)
        .reset_index(name="global_freq")
    )

    # --- 2. Remove globally common ingredients
    common_ingredients = global_freq.query("global_freq > @global_threshold")["ingredients"]
    df = df[~df["ingredients"].isin(common_ingredients)]

    # --- 3. Count remaining ingredient occurrences by continent
    counts = (
        df.groupby(["continent", "ingredients"])
        .size()
        .reset_index(name="count")
    )

    # --- 4. Select top N per continent
    top_by_continent = (
        counts.sort_values(["continent", "count"], ascending=[True, False])
        .groupby("continent", group_keys=False)
        .head(top_n)
        .sort_values("count", ascending=False)
    )

    # --- 5. Ensure continents appear consistently in sorted order
    continent_order = (
        df["continent"]
        .value_counts()
        .index.tolist()
    )
    top_by_continent["continent"] = pd.Categorical(
        top_by_continent["continent"], categories=continent_order, ordered=True
    )

    # --- 6. Plot: 2 columns layout for readability
    continents = top_by_continent["continent"].unique()
    n = len(continents)

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(13, 4 * nrows),
        sharey=False
    )

    axes = np.array(axes).reshape(-1)

    for ax, continent in zip(axes, continents):
        subset = (
            top_by_continent[top_by_continent["continent"] == continent]
            .sort_values("count", ascending=True)
        )

        sns.barplot(
            data=subset,
            x="count", y="ingredients",
            ax=ax, palette="viridis"
        )
        ax.set_title(continent, fontsize=12, fontweight="bold")
        ax.set_xlabel("Occurrences")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)
        ax.invert_yaxis()

    # Hide empty subplots if any
    for j in range(len(continents), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top {top_n} characteristic ingredients by continent (excluding ingredients used in > {int(global_threshold*100)}% of recipes)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()
    return fig

