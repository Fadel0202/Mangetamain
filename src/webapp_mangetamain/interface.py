"main function of the stremlit app"
import logging

import ingredients_analyzer
import streamlit as st
import utils.filter_data as filter_data
from nutriscore_analyzer import (
    add_nutriscore_column,
    analyze_low_scores_with_health_label,
    correlation_matrix,
    filter_data_with_nutri,
    parse_nutrition,
    plot_nutriscore_comparison,
)
from recipe_complexity import (
    make_corr_heatmap_fig,
    make_pairplot_fig,
    make_univariate_figs,
)
from tag_analyzer import (
    create_tag_recipes_dataset,
    filter_tags_of_interest,
    get_general_tags_statistics,
    plot_tag_frequency_distribution,
    plot_tags_per_recipe_distribution,
    plot_top_tags,
    plot_top_tags_by_metric,
)

from local_food import plot_cuisine_distributions, plot_top_ingredients_by_continent

from utils.filter_data import recipes_clean, separate_foods_drinks

from webapp_mangetamain.load_config import recipe, recipe_rating

logger = logging.getLogger(__name__)


def render_nutriscore_tab():
    """Render the Nutriscore tab content in Streamlit."""
    st.header("Nutriscore")
    st.subheader("Overview of nutritional scoring system")
    st.markdown(
        """
        ### Nutriscore

        We want to evaluate whether the **tags provided by users** are truly correlated
        with the **nutritional values** of the recipes.

        The **Nutriscore**, used in several European countries, provides a quick and simple way
        to assess the nutritional quality of a product based on data such as **calories, proteins, sugar, fat, salt**, etc.
        👉 [Reference](https://docs.becpg.fr/fr/utilization/score5C.html)

        - Each nutrient receives a **score**.
        - Some are considered **positive** (proteins, fiber).
        - Others are considered **negative** (calories, fat, salt).
        - The **final Nutriscore** is calculated as:
        `Nutriscore = Positive Score – Negative Score`.

        The goal is to explore **what relationships** can be extracted between the Nutriscore and
        other variables (user tags, nutritional values).
        """
    )
    # ------------------------
    # Correlation matrix
    # ------------------------
    st.markdown("""
    First, we can observe the different Nutri-Score values and their correlations with each other.
    We already notice that certain categories emerge: the correlations are stronger between calories, sugar, and fat. These negative values are what primarily lower the score.
    """)
    nutrition_df = parse_nutrition(recipe)
    st.subheader("Correlation Matrix of Nutrients")
    st.pyplot(correlation_matrix(nutrition_df))

    # ------------------------
    # Compare with 'healthy' tag
    # ------------------------
    filtered_df = filter_data_with_nutri(nutrition_df)
    scored_df = add_nutriscore_column(filtered_df)

    st.subheader("Comparison with 'health' tagged recipes")
    plot_nutriscore_comparison(scored_df, recipe)
    st.markdown("""
        **Something is wrong!**

        We should only see orange "health tag" bars in **categories A and B**.
        However, we observe that there are also many in **categories D and E**.
        While the proportion is lower in D, there are actually **more health tags in category E than there are recipes in category E**.

        This discrepancy could be explained by:
        - Incorrectly entered values,
        - The fact that **foods and drinks** use different Nutri-Score calculations,
        - Or the inability to verify whether the data is standardized (e.g., per 100g).
        """)
    # ------------------------
    # Drinks vs Foods
    # ------------------------
    st.markdown(
        """
        ## Drinks / Foods
        Note that our dataset includes both drinks and foods.
        However, the calculation of the Nutri-Score for drinks differs significantly from that for foods.
        Therefore, we focus exclusively on food items, filtering our data and recommendations accordingly.
        """
    )
    food_recipes, drink_recipes = separate_foods_drinks(recipe)
    st.subheader("Recipe Statistics")
    st.write(f"**Food recipes:** {len(food_recipes)}")
    st.write(f"**Drink recipes:** {len(drink_recipes)}")
    food_nutrition_df = nutrition_df.loc[food_recipes.index]
    filtered_df_food = filter_data_with_nutri(food_nutrition_df)
    scored_df_food = add_nutriscore_column(filtered_df_food)
    plot_nutriscore_comparison(scored_df_food, food_recipes)
    st.markdown("""
        **Something is still wrong!**

        We are seen the same patern as before. Even when we remove drinks from recipes.

        So this discrepancy could be explained by:
        - Incorrectly entered values,
        - Whether the data is standardized (e.g., per 100g).
        """)
    analyze_low_scores_with_health_label(recipe_df=food_recipes, nutrition_df=scored_df_food)

def render_tags_tab():
    """Render the Tags tab content."""
    st.header("Tags Analysis")
    st.subheader("Food categorization and labeling")
    # ========================
    # General Statistics
    # ========================
    st.markdown("---")
    st.write("### General Tag Statistics")
    stats = get_general_tags_statistics(recipe_rating)
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recipes", f"{stats['total_recipes']:,}")
    col2.metric("Unique Tags", stats['total_unique_tags'])
    col3.metric("Total Tags", f"{stats['total_tags']:,}")
    col4.metric("Avg Tags/Recipe", f"{stats['tags_per_recipe_mean']:.2f}")
    # All statistics in order
    with st.expander("View Complete Statistics"):
        st.write("#### Recipe-Level Statistics")
        st.write(f"- **Total recipes analyzed:** {stats['total_recipes']:,}")
        st.write(f"- **Average tags per recipe:** {stats['tags_per_recipe_mean']:.2f}")
        st.write(f"- **Median tags per recipe:** {stats['tags_per_recipe_median']:.0f}")
        st.write(f"- **Min tags per recipe:** {stats['tags_per_recipe_min']:.0f}")
        st.write(f"- **Max tags per recipe:** {stats['tags_per_recipe_max']:.0f}")
        st.write("#### Tag-Level Statistics")
        st.write(f"- **Total unique tags:** {stats['total_unique_tags']}")
        st.write(f"- **Total tags (with duplicates):** {stats['total_tags']:,}")
        st.write(f"- **Average occurrences per tag:** {stats['avg_tags_general']:.2f}")
        st.write("#### Distribution Details")
        st.write("**Tags per recipe (detailed):**")
        st.write(stats['tags_per_recipe_stats'])
        st.write("**Tag frequency (detailed):**")
        st.write(stats['tag_counts_stats'])
        st.write("#### Top 20 Most Frequent Tags")
        st.write(stats['top_20_tags'])

    # ========================
    # Visualizations
    # ========================
    st.markdown("---")
    st.write("### Tag Distributions")

    # Graph 1: Tag frequency first
    st.write("#### Tag Frequency Distribution")
    tag_counts = recipe_rating['tags_parsed'].explode().value_counts()
    plot_tag_frequency_distribution(tag_counts)
    st.info("Power law distribution: few tags used very frequently, many tags used rarely")

    # Graph 2: Top 20
    st.write("#### Top 20 Most Frequent Tags")
    plot_top_tags(tag_counts, top_n=20)
    st.info("Generic organizational tags dominate (preparation, time-to-make, course)")

    # Graph 3: Distribution per recipe
    st.write("#### Distribution of Tags per Recipe")
    plot_tags_per_recipe_distribution(recipe_rating)
    st.info("Most recipes have between 13-22 tags. Mean: 17.9, Median: 17.0")

    # ========================
    # Tags Analysis by Metrics
    # ========================
    st.markdown("---")
    st.write("### Tags Analysis by Metrics")
    tag_stats, _ = create_tag_recipes_dataset(recipe_rating, min_recipes_per_tag=50)
    st.info(f"Analyzing **{len(tag_stats)}** tags with at least 50 recipes")
    tags_of_interest = filter_tags_of_interest(tag_stats)
    st.success(f"Found **{len(tags_of_interest)}** tags of interest across **{tags_of_interest['category'].nunique()}** categories")
    # Show the 37 tags of interest
    with st.expander("View the tags of interest"):
        for category in sorted(tags_of_interest['category'].unique()):
            cat_tags = tags_of_interest[tags_of_interest['category'] == category]['tag'].tolist()
            st.write(f"**{category}** ({len(cat_tags)} tags): {', '.join(cat_tags)}")
    st.write("#### Top Tags by Number of Recipes")
    plot_top_tags_by_metric(tag_stats, metric='n_recipes', top_n=20)

    # ========================
    # Best Tags
    # ========================
    st.markdown("---")
    st.write("### 🏆 Best Tags by Criteria")
    # Simple best tags without function
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Fastest (by avg time):**")
        fastest = tags_of_interest.nsmallest(5, 'avg_minutes')[['tag', 'category', 'avg_minutes', 'n_recipes']]
        st.dataframe(fastest, hide_index=True)

        st.write("**Simplest (fewest ingredients):**")
        simplest = tags_of_interest.nsmallest(5, 'avg_ingredients')[['tag', 'category', 'avg_ingredients', 'n_recipes']]
        st.dataframe(simplest, hide_index=True)

    with col2:
        st.write("**Most Popular:**")
        popular = tags_of_interest.nlargest(5, 'n_recipes')[['tag', 'category', 'n_recipes', 'avg_minutes']]
        st.dataframe(popular, hide_index=True)
        st.write("**Fewest Steps:**")
        easy = tags_of_interest.nsmallest(5, 'avg_steps')[['tag', 'category', 'avg_steps', 'n_recipes']]
        st.dataframe(easy, hide_index=True)


def render_ingredient_tab():
    """
    ingredients_exploded: DataFrame avec colonnes ['id','ingredients'] (déjà normalisées)
    """
    st.header("Ingredients")

    # === 1) Statistiques globales ===
    n_recettes = filter_data.ingredients_exploded["id"].nunique()
    n_ingredients_uniques = filter_data.ingredient_counts.shape[0]
    mean_ingredients_per_recipe = (
        filter_data.ingredients_exploded.groupby("id")["ingredients"].nunique().mean()
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique ingredients", f"{n_ingredients_uniques:,}")
    col2.metric("Recipes", f"{n_recettes:,}")
    col3.metric("Avg. ingredients per recipe", f"{mean_ingredients_per_recipe:.2f}")

    st.divider()
    # ===== 2) Distribution & résumé =====
    st.subheader("Distribution of the number of ingredients per recipe")
    st.pyplot(ingredients_analyzer.plot_ingredient_per_recette(filter_data.ingredients_exploded))
    st.markdown("""
    The number of ingredients per recipe generally ranges between **5 and 12**, with a peak around **8 ingredients**.
    This indicates that most recipes are **moderately complex**: not extremely simple, but not overly elaborate either.

    The distribution is **right-skewed**, meaning:
    - Very simple recipes with few ingredients are **less common**,
    - Very complex recipes (15+ ingredients) do exist but are **rare**, likely corresponding to festive or professional dishes.

    **Key Insight:**
    Most home cooking tends to use a **moderate number of ingredients**, balancing simplicity and flavor.
    """)

    st.subheader("Ingredient Frequency Distribution")
    ingredient_counts = filter_data.ingredient_counts
    st.dataframe(ingredients_analyzer.summarize_ingredient_stats(ingredient_counts))
    st.pyplot(ingredients_analyzer.plot_ingredient_distribution(ingredient_counts))
    st.markdown("""
    The ingredient frequency follows a **long-tail distribution**:
    - A **small set of ingredients** (e.g., *salt, butter, sugar, onion*) appears extremely frequently,
    - While **the majority of ingredients** appear only occasionally.

    Using a log scale makes this pattern clear:
    most ingredients are **rare**, while a few are **universal pantry staples**.

    **Key Insight:**
    This confirms that cooking relies on a **shared basic pantry**, enriched with **regional or personal variations**.
    """)

    st.subheader("Most Frequent Ingredients")
    top_n = st.slider("Display the top N most frequent ingredients", 10, 100, 30, 5)
    st.pyplot(ingredients_analyzer.make_top_ingredients_bar_fig(ingredient_counts, top_n))
    st.markdown("""
    The top ingredients include:
    **salt, butter, sugar, onion, eggs, olive oil, flour, garlic, milk, pepper**.

    These represent:
    - **Base seasoning:** salt, pepper, garlic
    - **Core fats:** butter, olive oil
    - **Structural staples:** flour, eggs, milk

    Because these ingredients appear **almost everywhere**, they are **not useful** for distinguishing:
    - Culinary styles
    - Dietary constraints
    - Regional cuisines

    **Therefore, we exclude these 'omnipresent' ingredients in further analysis.**
    """)

    # ===== 3) Fenêtre de fréquence =====
    st.subheader("Frequency Window Selection")
    st.markdown("""
    To extract **meaningful ingredient patterns**, we restrict the analysis to ingredients appearing
    **often enough to be relevant**, but **not so frequently that they appear everywhere**.

    Example window:
    min_count = 100
    max_count = 5000
    This removes:
    - **Rare ingredients** (not statistically meaningful)
    - **Universal staples** (not informative)

    **Key Insight:**
    This step isolates the **ingredients that define cooking identity**, such as:
    regional flavors, cuisine families, or dietary patterns.
    """)


    min_count = st.number_input(
        "min_count (exclude very rare ingredients)",
        1, int(ingredient_counts["count"].max()), 200
    )
    use_max = st.checkbox("Limit very common ingredients (max_count)", value=True)
    default_max = 5000
    max_count = st.number_input(
        "max_count", min_count, int(ingredient_counts["count"].max()),
        default_max, step=50
    ) if use_max else None

    kept_counts = filter_data.filter_counts_window(
        ingredient_counts, min_count=min_count, max_count=max_count
    )

    st.caption(
        f"→ {len(kept_counts):,} ingredients kept after filtering "
        f"(representing {kept_counts['count'].sum():,} total usages)"
    )

    st.subheader("Ingredient Neighborhood — Similarity-Based Pairings")
    st.markdown("""
    By selecting an ingredient, we compute the ingredients that **co-occur most often with it**
    (using **Jaccard similarity** or **conditional probability P(B|A)**).

    This reveals:
    - **Flavor pairings**
    - **Substitution patterns**
    - **Cooking contexts**

    Example:
    Choosing **‘low-fat milk’** suggests ingredients such as:
    `whole wheat flour`, `egg whites`, `canola oil`, `cooking spray`.

    ## Interpretation:
    These combinations point to **healthy baking / breakfast cuisine**.

    This feature is especially useful for:

    - Suggesting substitutes
    - Understanding cuisine profiles
    - Exploring ingredient roles in recipes
    """)
    c1, c2 = st.columns([2, 1])
    with c1:
        focus = st.selectbox("Select an ingredient", sorted(filter_data.co_occurrence.columns.to_list()))
    with c2:
        k = st.slider("Top K", 5, 40, 15)

    min_co_focus = st.slider("Minimum co-occurrence (|A∩B|) for focus", 1, 200, 20)
    metric = st.radio("Mesure", ["Jaccard"], horizontal=True)

    if metric == "Jaccard":
        assoc = ingredients_analyzer.top_cooccurrences_for(focus, filter_data.jaccard, filter_data.co_occurrence, k=k, min_co=min_co_focus)
        x_field, title = "score", f"Top neighbors Jaccard avec '{focus}'"


    st.pyplot(ingredients_analyzer.make_association_bar_fig(assoc, title, x=x_field))
    st.dataframe(assoc)

def render_complexity_tab():
    df = recipes_clean
    """Render the Complexity tab content in Streamlit."""
    st.header("Complexity")
    st.markdown(
        """
        Explore how **time**, **number of steps**, and **number of ingredients** relate to each other.
        We first look at each feature individually, then we inspect their relationships (pairplot + correlation matrix).
        """
    )

    st.subheader("Univariate exploration")
    feature = st.radio(
        "Choose a feature:",
        ["minutes", "n_steps", "n_ingredients"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    hist_fig, box_fig = make_univariate_figs(df, feature, hue=("kind" if "kind" in df.columns else None))
    with col1:
        st.pyplot(hist_fig)
    with col2:
        st.pyplot(box_fig)
    if feature == "minutes":
        st.markdown("""
    ###Univariate Analysis — Preparation Time (`minutes`)

    The distribution of preparation time is **highly skewed**.
    Most recipes take **less than 60 minutes**, but a small number of slow-cook or resting-time recipes
    extend the distribution to several hours.

    -> This creates a **long right tail**, making the raw time scale misleading.
    To address this, we use **`log(minutes)`** in visualizations and correlations, which **compresses extreme values**
    and reveals the underlying structure of typical recipes.
    """)

    elif feature == "n_steps":
        st.markdown("""
    ### Univariate Analysis — Number of Steps (`n_steps`)

    The number of steps is more **tightly distributed**.
    Most recipes have **between 5 and 15 steps**.

    - Very short recipes (1–3 steps) usually correspond to **simple preparations**.
    - Recipes with **more than 20 steps** tend to be **complex meals, baking workflows, or multi-phase preparations**.

    -> `n_steps` is a strong indicator of **procedural complexity** in a recipe.
    """)

    else:  # n_ingredients
        st.markdown("""
    ### Univariate Analysis — Number of Ingredients (`n_ingredients`)

    Most recipes use **between 5 and 12 ingredients**, which corresponds to a moderate level of complexity.

    - Few-ingredient recipes are often **simple bases or minimal dishes**.
    - High-ingredient recipes usually indicate **rich, festive, or slow-cooked dishes**.

    -> While useful, this metric is **less strongly tied to complexity** than the number of steps.
    """)

    st.subheader("Relationships between features")
    features_rel = ["log_minutes", "n_steps", "n_ingredients"]
    pair_fig = make_pairplot_fig(df, features_rel, hue=("kind" if "kind" in df.columns else None))
    st.pyplot(pair_fig)
    st.markdown("""
    ### Relationships Between Complexity Features

    The pairplot highlights **positive relationships** among the three dimensions:

    - **`n_steps` ↔ `n_ingredients`** — **Strongest relationship**
    → Recipes that require more steps tend to involve **more ingredients**.

    - **`log(minutes)` ↔ `n_steps`** — Moderate relationship
    → Longer recipes often have more steps, but this is not always the case:
        some recipes have **long passive waits** (resting, slow cooking), increasing time without adding complexity.

    - **`log(minutes)` ↔ `n_ingredients`** — Weaker relationship
    → A dish may have many ingredients *without* being time-consuming (e.g., composed salads, marinades).

    -> **Key Insight:**
    The **number of steps** is the most reliable indicator of **practical complexity**, since it measures effort more directly.
    """)

    st.subheader("Correlation matrix")
    corr_fig = make_corr_heatmap_fig(df, features_rel, "Correlation (log_minutes, n_steps, n_ingredients)")
    st.pyplot(corr_fig)
    st.markdown("""
    ### Correlation Matrix — Summary

    | Variable Pair | Correlation | Interpretation |
    |--------------|------------|----------------|
    | `n_steps` ↔ `n_ingredients` | **0.40 – 0.45** | More steps usually imply more ingredients. |
    | `log(minutes)` ↔ `n_steps` | **0.30 – 0.35** | Longer recipes tend to require more steps, except when time is passive. |
    | `log(minutes)` ↔ `n_ingredients` | **0.25 – 0.35** | Ingredient count contributes to complexity but not strongly to cooking time. |

    **Overall Conclusion:**

    - **`n_steps` is the strongest single proxy for recipe complexity.**
    - **Time must be log-transformed** to be interpreted properly.
    - **Ingredient count** adds context but is secondary.

    In short, recipe complexity is **structural** (how much you *do*), not just **temporal** (how long it cooks).
    """)


def render_other_tab():
    """Render the Other tab content."""
    st.header("Other")
    st.subheader("Additional section placeholder")
    st.info("This tab can be customized for additional features.")


def render_local_food_tab():
    """
    Streamlit Tab: Analysis of cuisines by continent.
    """
    st.header("Cuisine Analysis by Continent")

    # === Load data ===
    recipes_with_continent = filter_data.recipes_with_continent
    ingredient_and_continent = filter_data.ingredient_and_continent

    st.caption(f"Total number of recipes: **{len(recipes_with_continent):,}**")

    # === 1) Distribution plots ===
    st.subheader("Distribution of recipe complexity by continent")
    st.pyplot(plot_cuisine_distributions(recipes_with_continent))

    st.markdown("""
    **Interpretation:**
    The boxplots show clear regional differences in recipe complexity:
    - **North American** and **Oceania** cuisines tend to have shorter preparation times and fewer steps.
    - **Asian** and **European** cuisines show a wider spread, indicating more diversity in cooking times and methods.
    - **Africa & Middle East** and **Central & South America** often use more ingredients per recipe, suggesting richer flavor profiles.  
    Overall, Asian and European cuisines appear the most complex, while Oceania recipes are the simplest.
    """)

    # === 2) Typical ingredients ===
    st.subheader("🥕 Typical ingredients by continent")

    top_n = st.slider(
        "Number of ingredients to display per continent",
        min_value=5, max_value=30, value=10, step=1
    )

    threshold = st.slider(
        "Omnipresence threshold (exclude ingredients appearing in more than X% of recipes)",
        min_value=0.05, max_value=1.0, value=0.30, step=0.05
    )

    st.caption(
        f"Ingredients appearing in more than **{int(threshold * 100)}%** of recipes "
        f"are excluded to avoid global staples like *salt*, *water*, or *sugar*."
    )

    st.pyplot(
        plot_top_ingredients_by_continent(
            ingredient_and_continent,
            top_n=top_n,
            global_threshold=threshold
        )
    )

    st.markdown("""
    **Interpretation:**
    The chart highlights the **signature ingredients** of each continent:
    - **Asia:** soy sauce, sesame oil, rice wine, and curry leaves dominate, emphasizing umami-rich flavors.
    - **Europe:** strong presence of pasta, cheese, and tomato-based products, typical of Mediterranean and Italian influences.
    - **North America:** a prevalence of processed products (cheese blends, corn syrup), reflecting industrialized cooking habits.
    - **Central & South America:** ingredients like *adobo*, *tomatillos*, and *queso fresco* reveal a spicy and colorful cuisine.
    - **Africa & Middle East:** ingredients such as *harissa*, *rose water*, and *coriander seed* reflect deep spice traditions.
    - **Oceania:** recipes often reuse European bases but with regional variations like *macadamia* and *golden syrup*.

    Overall, each continent shows clear culinary identity once global ingredients are filtered out.
    """)

    # === 3) User tip ===
    st.info(
        "💡 Tip: Try lowering the omnipresence threshold (e.g. 10–20%) to highlight truly distinctive regional ingredients."
    )


def main():
    """Main function to run the MangeTaMain Dashboard."""
    # App title
    st.title("MangeTaMain Dashboard")

    # Create tabs
    tab1, tab2, tab3, tab4 , tab5 = st.tabs(["Nutriscore", "Tags", "Ingredient", "Complexity", "Local Food"])

    with tab1:
        render_nutriscore_tab()

    with tab2:
        render_tags_tab()

    with tab3:
        render_ingredient_tab()

    with tab4:
        render_complexity_tab()

    with tab5:
        render_local_food_tab()


if __name__ == "__main__":
    main()
