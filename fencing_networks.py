import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo

    return mo, pd


@app.cell
def _():
    rt_path= "C:/Users/bensa/OneDrive/Documents/Important Docs/Portfolio/Fencing/olympic fencing dataset/Womens foil olympics/"
    return (rt_path,)


@app.cell
def _(pd, rt_path):
    bouts = pd.read_csv(rt_path + "all_womens_foil_bout_data_May_13_2021_cleaned.csv")
    return (bouts,)


@app.cell
def _(bouts):
    bouts.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # questions
    - is it better to store the network as a matrix or a network object, is there a difference
    - how should i weight the network, especially for repeat matches.
    - i should check if there are any repeat matches [there are]
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(bouts):
    # Compute directed and undirected pair matches to see if there are repeat pairs
    import numpy as np

    # Directed pairs (exact same fencer_ID and opp_ID)
    directed_counts = bouts.groupby(['fencer_ID', 'opp_ID']).size().reset_index(name='match_count')
    repeat_directed = directed_counts[directed_counts['match_count'] > 1]

    # Undirected pairs (treating fencer A vs fencer B the same as fencer B vs fencer A)
    bouts_undirected = bouts.copy()
    bouts_undirected['fencer_min'] = np.minimum(bouts['fencer_ID'], bouts['opp_ID'])
    bouts_undirected['fencer_max'] = np.maximum(bouts['fencer_ID'], bouts['opp_ID'])

    undirected_counts = bouts_undirected.groupby(['fencer_min', 'fencer_max']).size().reset_index(name='match_count')
    repeat_undirected = undirected_counts[undirected_counts['match_count'] > 1]
    return (np,)


@app.cell
def _(bouts, np):
    # Identify the winner and loser for each bout
    bouts_with_loser = bouts.copy()
    bouts_with_loser["loser_ID"] = np.where(
        bouts_with_loser["winner_ID"] == bouts_with_loser["fencer_ID"],
        bouts_with_loser["opp_ID"],
        bouts_with_loser["fencer_ID"],
    )

    # Group by winner and loser to get win counts
    win_counts = (
        bouts_with_loser.groupby(["winner_ID", "loser_ID"])
        .size()
        .reset_index(name="wins")
    )

    # Get all unique fencer IDs to construct a square matrix
    all_fencers = sorted(
        list(set(bouts["fencer_ID"]).union(set(bouts["opp_ID"])))
    )

    # Pivot and reindex to create the square adjacency matrix
    adjacency_matrix = (
        win_counts.pivot(index="winner_ID", columns="loser_ID", values="wins")
        .fillna(0)
        .reindex(index=all_fencers, columns=all_fencers, fill_value=0)
        .astype(int)
    )

    adjacency_matrix
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
