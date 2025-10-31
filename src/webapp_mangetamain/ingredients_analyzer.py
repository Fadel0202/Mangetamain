import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_ingredient_per_recette(ingredients_exploded: pd.DataFrame):
    counts_per_recipe = ingredients_exploded.groupby("id")["ingredients"].nunique()
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    sns.histplot(counts_per_recipe, bins=30, kde=True, color="skyblue", ax=ax1)
    ax1.set_xlabel("Number of ingredients per recipe")
    ax1.set_ylabel("Number of recipes")
    return fig1


def plot_ingredient_distribution(ingredient_counts: pd.DataFrame):
    """
    Return matplotlib figure: log-scaled boxplot of ingredient frequencies.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(np.log1p(ingredient_counts["count"].values), bins=50, kde=True, color="salmon", ax=ax)
    ax.set_title("Log distribution of ingredient frequencies")
    ax.set_xlabel("log(1 + frequency)")
    return fig


def summarize_ingredient_stats(ingredient_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and return quantile summary (describe + percentiles).
    """
    print(ingredient_counts.head)
    quantiles = ingredient_counts["count"].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
    return pd.DataFrame(quantiles).T.round(2)


def make_top_ingredients_bar_fig(ingredient_counts: pd.DataFrame, top_n: int = 30) -> plt.Figure:
    """
    Affiche un barplot des ingrédients les plus fréquents.

    Paramètres :
        ingredient_counts : DataFrame avec colonnes ['ingredient', 'count']
        top_n : nombre d'ingrédients à afficher
    """
    top_df = ingredient_counts.nlargest(top_n, "count")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    sns.barplot(
        data=top_df,
        y="ingredients",
        x="count",
        ax=ax,
        palette="viridis"
    )
    ax.set_title(f"Top {top_n} most frequent ingredients", fontsize=13, pad=10)
    ax.set_xlabel("Number of occurrences")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def make_counts_boxplot_fig(ingredient_counts: pd.DataFrame) -> plt.Figure:
    """
    Boxplot log des fréquences d'ingrédients pour visualiser la distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=np.log1p(ingredient_counts["count"]), ax=ax, color="lightblue")
    ax.set_title("Log distribution of ingredient counts")
    ax.set_xlabel("log(1 + count)")
    plt.tight_layout()
    return fig


def top_cooccurrences_for(ingredient, jaccard, co_occurrence, k=15, min_co=20):
    ing = ingredient.lower().strip()
    if ing not in jaccard.index:
        print(f"'{ingredient}' n'existe pas dans la matrice.")
        return pd.DataFrame()

    # ligne Jaccard + co-occurrence
    s = jaccard.loc[ing].copy()
    co = co_occurrence.loc[ing].copy()

    # filtrage
    mask = co >= min_co
    s = s[mask].drop(index=ing, errors="ignore")
    co = co[mask].drop(index=ing, errors="ignore")

    # top k
    out = (
        pd.DataFrame({"other": s.index, "score": s.values, "co": co.loc[s.index].values})
        .sort_values("score", ascending=False)
        .head(k)
    )
    return out

def make_association_bar_fig(df_pairs: pd.DataFrame, title: str, x: str = "lift") -> plt.Figure:
    """
    Barplot horizontal des associations (x = 'lift' ou 'P(B|A)').
    """
    if df_pairs.empty:
        fig, ax = plt.subplots(figsize=(6, 0.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    df_plot = df_pairs.sort_values([x, "co"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=df_plot, y="other", x=x, ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel("Ingredient")
    ax.set_title(title)
    fig.tight_layout()
    return fig


