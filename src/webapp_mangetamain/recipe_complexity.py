import pandas as pd
import seaborn as sns # tyoe: ignore
import matplotlib.pyplot as plt

def make_univariate_figs(df: pd.DataFrame, feature: str, hue: str | None = None):
    """
    Retourne (hist_fig, box_fig) pour une feature.
    """
    data = df[feature]
    # hist
    hist_fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.histplot(data, bins=40, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    hist_fig.tight_layout()

    # box
    box_fig, ax2 = plt.subplots(figsize=(5, 3.5))
    if hue and hue in data.columns:
        sns.boxplot(data=data, x=hue, y=feature, ax=ax2)
        ax2.set_title(f"{feature} by {hue}")
        ax2.set_xlabel(hue)
        ax2.set_ylabel(feature)
    else:
        sns.boxplot(x=data, ax=ax2)
        ax2.set_title(f"Boxplot of {feature}")
        ax2.set_xlabel(feature)
    box_fig.tight_layout()

    return hist_fig, box_fig


def make_pairplot_fig(df: pd.DataFrame, features: list[str], hue: str | None = None):
    """Retourne la figure du pairplot."""
    data = df[features]

    g = sns.pairplot(
        data,
        vars=features,
        diag_kind="kde",
        hue=(hue if hue and hue in data.columns else None),
        corner=True,
        plot_kws=dict(alpha=0.5, s=15),
    )
    return g.figure

def make_corr_heatmap_fig(df: pd.DataFrame, features: list[str], title: str = "Correlation matrix"):
    """Retourne la figure de la heatmap de corrélation."""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
