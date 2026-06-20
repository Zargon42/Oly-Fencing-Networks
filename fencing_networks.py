import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
