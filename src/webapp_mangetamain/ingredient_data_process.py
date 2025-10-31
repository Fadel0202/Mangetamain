import numpy as np
import pandas as pd
import utils.filter_data as filter_data
from pathlib import Path


def generate_matrix():
    top_ingredients = filter_data.ingredient_counts
    ingredients_exploded = filter_data.ingredients_exploded
    relevant_ingredients = top_ingredients[(top_ingredients > 100)  & (top_ingredients < 5000) ]
    print(relevant_ingredients)


    relevant_ingredients_exploded = ingredients_exploded[ingredients_exploded["ingredients"].str.lower().isin(relevant_ingredients.index)]
    relevant_ingredients_exploded = relevant_ingredients_exploded[["id", "ingredients"]]
    print(relevant_ingredients_exploded.head())

    bin_df = (
        relevant_ingredients_exploded
        .assign(val=1)
        .pivot_table(index="id", columns="ingredients", values="val", fill_value=0)
    )

    co_occurrence = bin_df.T.dot(bin_df)

    diag = np.diag(co_occurrence)
    union = (diag[:, None] + diag[None, :] - co_occurrence.values)

    jaccard = pd.DataFrame(
        co_occurrence.values / np.where(union == 0, 1, union),
        index=co_occurrence.index,
        columns=co_occurrence.columns)


    # 1) Renommer les axes pour éviter le conflit lors de reset_index
    jacc = jaccard.copy()
    jacc.index.name = "ing_a"
    jacc.columns.name = "ing_b"

    # 2) Enlever la diagonale (self-similarity)
    np.fill_diagonal(jacc.values, 0.0)

    min_co = 10
    co = (bin_df.T @ bin_df).astype(int)         # matrice de co-occurrence brute
    co.index.name = "ing_a"
    co.columns.name = "ing_b"
    mask = co >= min_co
    jacc_filt = jacc.where(mask, other=0.0)

    pairs = (
        jacc_filt.stack()
                .reset_index(name="score")      # plus de conflit de nom
                .query("ing_a < ing_b and score > 0")
                .sort_values("score", ascending=False)
    )

    print(len(relevant_ingredients))
    print(len(pairs))



    path = Path("artifacts")
    path.mkdir(exist_ok=True)

    print(co_occurrence.head())
    print(jaccard.head())

    co_occurrence.to_csv("artifacts/co_occurrence.csv")
    jaccard.to_csv("artifacts/jaccard.csv")





