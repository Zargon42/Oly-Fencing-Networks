import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rankings from pairwise comparisons
    Here we explore how to extract hidden rankings from pairwise comparisons, e.g. games between teams in sport.


    ### Credit:
    This notebook is heavily based off the code of Professor Caterina De Bacco (https://www.cdebacco.com/#about).
    The dataset used was collected by Andrew Fischl, otherwise known as CyrusOfChaos on Instagram and YouTube
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import networkx as nx

    return np, nx, pd


@app.cell
def _():
    import sys
    sys.path.insert(0, "src")
    sys.path.append('../../../src/')
    import tools as tl
    import plot as viz
    from plot import BLACK
    import io as io
    import ranking_tools.springrank as sr
    import ranking_tools.bradley_terry as bt
    import ranking_tools.process_input_into_matrix as prcs


    return bt, sr


@app.cell
def _():
    import scipy.stats as st
    from sklearn.cluster import AffinityPropagation
    from matplotlib.lines import Line2D
    from adjustText import adjust_text
    from scipy.stats import pearsonr, spearmanr
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns

    return (
        AffinityPropagation,
        LabelEncoder,
        adjust_text,
        pearsonr,
        sns,
        spearmanr,
        st,
    )


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colormap = plt.cm.tab10
    colors = {i: colormap(i) for i in range(20)}
    return colors, plt


@app.cell
def _():
    import cv_tools as cvtl


    return


@app.cell
def _():
    # pdf reading
    import re
    import pdfplumber
    import os

    return (os,)


@app.cell
def _():
    import requests
    from io import StringIO


    return StringIO, requests


@app.cell
def _(np):
    seed = 10
    prng = np.random.RandomState(seed)
    return


@app.cell
def _(plt):
    # Override the custom library's font choice to a system-safe font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Download code
    - [SpringRank](https://github.com/LarremoreLab/SpringRank/blob/master/springrank/springrank.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Import data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## import fencing data
    """)
    return


@app.cell
def _(pd):
    bouts_pool = pd.read_csv("CoC dataset/Pools.csv")
    bouts_DE = pd.read_csv("CoC dataset/DEs.csv")
    #athletes = pd.read_csv("CoC dataset/Athletes.csv")
    return bouts_DE, bouts_pool


@app.cell
def _(bouts_pool):
    bouts_pool.head()
    return


@app.cell
def _(bouts_DE, bouts_pool, pd):
    # Combine pool and DE bouts
    all_fencing_bouts = pd.concat([bouts_pool, bouts_DE], ignore_index=True)

    # Drop rows with missing values in critical columns
    all_fencing_bouts = all_fencing_bouts.dropna(subset=['Fencer A', 'Fencer B', 'Winner']).copy()
    return (all_fencing_bouts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### import rankings
    """)
    return


@app.cell
def _(StringIO, pd, requests):
    def fie_ranking_to_csv(url, csv_path=None):
        """
        Download an FIE detailed ranking page and convert the ranking
        table to a CSV.

        Parameters
        ----------
        url : str
            FIE detailed-ranking URL.

        csv_path : str, optional
            Output CSV path. If None, creates a CSV next to the URL's
            inferred ranking name.

        Returns
        -------
        pandas.DataFrame
            Cleaned FIE ranking dataframe.
        """

        # ----------------------------------------------------------
        # 1. Download page like a normal browser
        # ----------------------------------------------------------

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://fie.org/",
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        # ----------------------------------------------------------
        # 2. Extract tables from downloaded HTML
        # ----------------------------------------------------------

        tables = pd.read_html(
            StringIO(response.text)
        )

        if not tables:
            raise ValueError(
                "No tables found on the FIE page."
            )

        # Find the ranking table rather than blindly assuming
        # it is table 0.
        ranking_table = None

        for table in tables:

            columns = [
                str(col).strip()
                for col in table.columns
            ]

            if (
                "Rank" in columns
                and "Name" in columns
            ):
                ranking_table = table.copy()
                break

        if ranking_table is None:
            raise ValueError(
                "Could not find the FIE ranking table."
            )

        df = ranking_table

        # ----------------------------------------------------------
        # 3. Clean column names
        # ----------------------------------------------------------

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        # Remove completely empty columns
        df = df.dropna(
            axis=1,
            how="all"
        )

        # ----------------------------------------------------------
        # 4. Clean Rank
        # ----------------------------------------------------------

        if "Rank" in df.columns:

            df["Rank"] = pd.to_numeric(
                df["Rank"],
                errors="coerce"
            ).astype("Int64")

            # Keep only actual ranking rows
            df = df[
                df["Rank"].notna()
            ]

        # ----------------------------------------------------------
        # 5. Clean numeric columns
        # ----------------------------------------------------------

        for col in df.columns:

            if col in ["Rank", "Name", "Nat."]:
                continue

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .replace("", pd.NA)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ----------------------------------------------------------
        # 6. Remove duplicates
        # ----------------------------------------------------------

        duplicate_columns = [
            col
            for col in ["Rank", "Name", "Nat."]
            if col in df.columns
        ]

        if duplicate_columns:

            df = df.drop_duplicates(
                subset=duplicate_columns,
                keep="first"
            )

        # ----------------------------------------------------------
        # 7. Sort by ranking
        # ----------------------------------------------------------

        if "Rank" in df.columns:

            df = df.sort_values(
                by="Rank"
            ).reset_index(drop=True)

        # ----------------------------------------------------------
        # 8. Generate CSV path if none supplied
        # ----------------------------------------------------------

        if csv_path is None:

            csv_path = "fie_ranking.csv"

        # ----------------------------------------------------------
        # 9. Save
        # ----------------------------------------------------------

        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Saved {len(df)} fencers "
            f"with {len(df.columns)} columns to:"
        )
        print(csv_path)

        return df

    return (fie_ranking_to_csv,)


@app.cell
def codenames():
    '''rankings = {
        "ME": ("E", "M"),
        "MF": ("F", "M"),
        "MS": ("S", "M"),
        "WE": ("E", "F"),
        "WF": ("F", "F"),
        "WS": ("S", "F"),
    }

    for code, (weapon, gender) in rankings.items():

        url = (
            "https://fie.org/athletes/detailed-ranking"
            f"?season=2026"
            f"&weapon={weapon}"
            f"&gender={gender}"
            f"&category=S"
            f"&type=I"
        )

        fie_ranking_to_csv(
            url,
            f"rankings/{code}-detailed-ranking-2026.csv"
        )'''
    return


@app.cell
def _(fie_ranking_to_csv, mo, os):
    # Check if all 6 FIE CSV files already exist on disk to prevent slow web requests
    _rankings_definitions = {
        "ME": ("E", "M"),
        "MF": ("F", "M"),
        "MS": ("S", "M"),
        "WE": ("E", "F"),
        "WF": ("F", "F"),
        "WS": ("S", "F"),
    }

    _missing_any_files = False
    for _code in _rankings_definitions.keys():
        _path = f"rankings/{_code}-detailed-ranking-2026.csv"
        if not os.path.exists(_path):
            _missing_any_files = True
            break

    if not _missing_any_files:
        # Skip download completely
        download_status_message = mo.md("✅ **FIE Rankings CSV files already exist on disk. Skipping slow web downloads.**")
    else:
        # Ensure directory exists
        os.makedirs("rankings", exist_ok=True)
    
        # Download ONLY the missing files
        for _code, (_weapon, _gender) in _rankings_definitions.items():
            _path = f"rankings/{_code}-detailed-ranking-2026.csv"
            if not os.path.exists(_path):
                _url = (
                    "https://fie.org/athletes/detailed-ranking"
                    f"?season=2026"
                    f"&weapon={_weapon}"
                    f"&gender={_gender}"
                    f"&category=S"
                    f"&type=I"
                )
                try:
                    fie_ranking_to_csv(_url, _path)
                except Exception as _e:
                    print(f"Error downloading {_code} from FIE: {_e}")
                
        download_status_message = mo.md("📥 **Completed: Missing FIE Ranking files downloaded and cached successfully!**")

    download_status_message

    return (download_status_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis
    """)
    return


@app.cell
def _(all_fencing_bouts, mo):
    class CustomDict(mo.ui.dictionary):
        @property
        def value(self):
            val = super().value
            p = val.get('primary')
            s = val.get('secondary')
            if s and s != 'None' and p != s:
                combined = f"{p} & {s}"
                if 'all_fencing_bouts' in globals():
                    all_fencing_bouts[combined] = all_fencing_bouts[p].astype(str) + " / " + all_fencing_bouts[s].astype(str)
                return combined
            return p

    # 1. Group / Subdivide Selector
    group_by_selector = CustomDict({
        'primary': mo.ui.dropdown(
            options=['Weapon', 'Age Category', 'Gender', 'Season', 'Host Country'],
            value='Weapon',
            label="Group/Compare Leagues By:"
        ),
        'secondary': mo.ui.dropdown(
            options=['None', 'Weapon', 'Age Category', 'Gender', 'Season', 'Host Country'],
            value='Gender',
            label="Subdivide By (Optional):"
        )
    })

    # 2. Model Selector
    model_selector = mo.ui.dropdown(
        options=['SpringRank', 'Bradley-Terry'],
        value='SpringRank',
        label="Select Ranking Model:"
    )

    # 3. Tuning parameters
    min_bouts_slider = mo.ui.slider(start=1, stop=20, step=1, value=5, label="Min Bouts Required")
    use_connectivity_filter = mo.ui.checkbox(label="Filter Isolated Events (Keep Largest Component)", value=True)

    mo.vstack([
        mo.md("### 📊 Fencing Analysis Control Panel"),
        mo.md("#### 1. Define Your Leagues"),
        mo.hstack([group_by_selector['primary'], group_by_selector['secondary']], justify="start"),
        mo.md("#### 2. Configure Model Settings"),
        mo.hstack([model_selector, min_bouts_slider, use_connectivity_filter], justify="start")
    ])
    return (
        group_by_selector,
        min_bouts_slider,
        model_selector,
        use_connectivity_filter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For now it is recomended to stick to the SpringRank model as bradley terry produces less clear analysis and visualisations.
    """)
    return


@app.cell
def computationcell(
    LabelEncoder,
    all_fencing_bouts,
    bt,
    group_by_selector,
    min_bouts_slider,
    model_selector,
    np,
    nx,
    pd,
    sr,
    use_connectivity_filter,
):
    # Declare reactive inputs
    group_col = group_by_selector.value
    model_choice = model_selector.value
    min_bouts = min_bouts_slider.value
    filter_connected = use_connectivity_filter.value

    fencer_scores_list = []
    league_depth_stats = []

    if 'all_fencing_bouts' in globals():
        # Filter out rows where the grouping column is missing
        valid_bouts = all_fencing_bouts.dropna(subset=[group_col])

        # Group the bouts by the selected category
        grouped_bouts = valid_bouts.groupby(group_col)

        for group_name, group_df in grouped_bouts:
            if len(group_df) < 50:
                continue

            # 1. Filter out inactive fencers
            all_fenders_in_group = pd.concat([group_df['Fencer A'], group_df['Fencer B']])
            fencer_appearance_counts = all_fenders_in_group.value_counts()
            active_fencers = fencer_appearance_counts[fencer_appearance_counts >= min_bouts].index

            filtered_df = group_df[
                group_df['Fencer A'].isin(active_fencers) & 
                group_df['Fencer B'].isin(active_fencers)
            ].copy()

            if len(filtered_df) < 20:
                continue

            # 2. NetworkX: Optional Largest Connected Component filter
            if filter_connected:
                G = nx.Graph()
                G.add_edges_from(zip(filtered_df['Fencer A'], filtered_df['Fencer B']))
                if len(G) == 0:
                    continue
                largest_cc = max(nx.connected_components(G), key=len)
                target_fencers = [f for f in active_fencers if f in largest_cc]
            else:
                target_fencers = list(active_fencers)

            if len(target_fencers) < 5:
                continue

            filtered_df = filtered_df[
                filtered_df['Fencer A'].isin(target_fencers) & 
                filtered_df['Fencer B'].isin(target_fencers)
            ]

            # Label Encoder setup
            le = LabelEncoder()
            le.fit(target_fencers)
            n_fencers = len(le.classes_)

            # Build adjacency matrix
            fencer_to_id = {name: i for i, name in enumerate(le.classes_)}
            u_arr = filtered_df['Fencer A'].map(fencer_to_id).values.astype(int)
            v_arr = filtered_df['Fencer B'].map(fencer_to_id).values.astype(int)
            win_a_mask = (filtered_df['Winner'] == filtered_df['Fencer A']).values

            A_mat = np.zeros((n_fencers, n_fencers))
            np.add.at(A_mat, (u_arr[win_a_mask], v_arr[win_a_mask]), 1)
            np.add.at(A_mat, (v_arr[~win_a_mask], u_arr[~win_a_mask]), 1)

            # 3. Fit Selected Ranking Model
            try:
                if model_choice == 'SpringRank':
                    sr_model = sr.SpringRank()
                    sr_model.fit(A_mat)
                    ranks = sr_model.ranks
                    beta = sr_model.get_beta()
                    depth = sr_model.depth
                    n_levels = sr_model.n_levels
                else: # Bradley-Terry
                    bt_model = bt.BradleyTerry()
                    bt_model.fit(A_mat, method='em')
                    ranks = bt_model.ranks
                    # Compute mock depth attributes for BT to avoid UI breaks
                    beta = 1.0 
                    depth = ranks.max() - ranks.min()
                    n_levels = len(np.unique(np.round(ranks, 1)))
            except Exception as e:
                continue

            # Store individual fencer scores
            for idx, fencer_name in enumerate(le.classes_):
                fencer_scores_list.append({
                    'Fencer': fencer_name,
                    'League': group_name,
                    'Calculated_Score': ranks[idx]
                })

            # Store global network metrics
            league_depth_stats.append({
                'League': group_name,
                'Fencers Count': n_fencers,
                'Bouts Count': len(filtered_df),
                'Beta (Predictability)': beta,
                'Depth': depth,
                'Number of Levels': n_levels
            })

    df_fencer_scores = pd.DataFrame(fencer_scores_list)
    df_league_depths = pd.DataFrame(league_depth_stats)
    return df_fencer_scores, df_league_depths, group_col, model_choice


@app.cell
def _(df_fencer_scores, df_league_depths, group_col, mo):
    mo.vstack([
        mo.md(f"### 1. Depth of Competition (Grouped by {group_col})"),
        mo.ui.table(
            df_league_depths.sort_values(by='Number of Levels', ascending=False),
            label="League Depth Summary"
        ),
        mo.md(f"### 2. Search Fencer Strengths"),
        mo.ui.table(
            df_fencer_scores.sort_values(by='Calculated_Score', ascending=False),
            label="Individual Fencer Rankings"
        )
    ])
    return


@app.cell
def _(df_fencer_scores, mo):
    # Create a dictionary of DataFrames dynamically split by your chosen dividers
    league_dfs = {}
    _tabs_dict = {}

    if 'df_fencer_scores' in globals() and not df_fencer_scores.empty:
        _unique_leagues = sorted(df_fencer_scores['League'].unique())
    
        for _league in _unique_leagues:
            # Filter and sort by calculated score
            _sub_df = df_fencer_scores[df_fencer_scores['League'] == _league].sort_values(
                by='Calculated_Score', ascending=False
            ).reset_index(drop=True)
        
            # Add a local rank column
            _sub_df.insert(0, 'Rank', range(1, len(_sub_df) + 1))
        
            # Save to the league dictionary
            league_dfs[_league] = _sub_df
        
            # Add to the visual tab selector
            _tabs_dict[_league] = mo.vstack([
                mo.md(f"### 🏆 {_league} Rankings ({len(_sub_df)} fencers)"),
                mo.ui.table(_sub_df[["Rank", "Fencer", "Calculated_Score"]])
            ])
        
        _tabs_display = mo.ui.tabs(_tabs_dict)
    else:
        _tabs_display = mo.md("Calculate scores to generate league rankings.")

    _tabs_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### interpretations
    #### Depth analysis
    - Beta = how good of a predictor the fencers' scores are in predicting the outcome of a given match, higher -> more predictable
    - Depth = How big of a gap there is between the best and the worst athletes in the league. Higher number -> more dominant vs weak fencers instead of everyone has a good chance of beating anybody
    - Number of Levels -> how many distinct "levels" exist at which everyone in a higher level as at least a 75% chance of winning against those in the lower level.
    -
    #### Fencer Score

    - Calculated_Score = numerical representation of the "strength" of the athlete
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fencer score distributions
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    adjust_text,
    df_fencer_scores,
    group_col,
    model_choice,
    np,
    plt,
    st,
):
    # Create summary statistics for the fencing leagues
    df_fencing_dist = df_fencer_scores.groupby('League')['Calculated_Score'].agg(['min', 'max', 'count', 'mean']).reset_index()

    n_leagues = len(df_fencing_dist)

    # Adjust plot size dynamically based on number of leagues
    _fig, _ax = plt.subplots(1, 1, figsize=(10, max(5, n_leagues * 1.5)))

    # Y axis tick indices
    _ys = np.arange(n_leagues, 0, -1)

    # Draw horizontal range lines
    _ax.hlines(_ys, xmin=df_fencing_dist['min'], xmax=df_fencing_dist['max'], alpha=0.5, color='grey', lw=2, zorder=1)

    # Plot min and max endpoints
    _ax.scatter(df_fencing_dist['max'], _ys, s=150, alpha=0.7, c='#1f77b4', edgecolors='black', label='Max Score', zorder=5)
    _ax.scatter(df_fencing_dist['min'], _ys, s=150, alpha=0.7, c='#aec7e8', edgecolors='black', label='Min Score', zorder=5)

    _fencers_to_label = []

    for _i, _league_name in enumerate(df_fencing_dist['League']):
        _league_fencers = df_fencer_scores[df_fencer_scores['League'] == _league_name]
        _y_val = _ys[_i]

        # Apply vertical jittering to prevent overlapping points
        _y_jittered = _y_val + st.t(df=6, scale=0.08).rvs(len(_league_fencers))

        # Extract fencer scores
        _scores = _league_fencers['Calculated_Score'].values

        # Cluster fencers into similarity bands
        if len(_scores) >= 3:
            _clustering = AffinityPropagation(random_state=5).fit(_scores.reshape(-1, 1))
            _clabels = _clustering.labels_
        else:
            _clabels = np.zeros(len(_scores), dtype=int)

        _cs = [plt.cm.tab20(_c % 20) for _c in _clabels]

        # Plot individual fencers
        _ax.scatter(_scores, _y_jittered, s=40, alpha=0.6, c=_cs, edgecolors='none', zorder=2)

        # Identify top 2 and bottom 1 fencers to label
        _sorted_fencers = _league_fencers.sort_values(by='Calculated_Score', ascending=False)
        if len(_sorted_fencers) > 0:
            # Top 2 fencers
            for _, _row in _sorted_fencers.head(2).iterrows():
                _fencers_to_label.append((_row['Calculated_Score'], _y_val + np.random.normal(0, 0.04), _row['Fencer']))
            # Bottom 1 fencer
            if len(_sorted_fencers) > 2:
                _row_bot = _sorted_fencers.iloc[-1]
                _fencers_to_label.append((_row_bot['Calculated_Score'], _y_val + np.random.normal(0, 0.04), _row_bot['Fencer']))

    # Add text labels and resolve overlap
    _ts = []
    for _x, _y, _lbl in _fencers_to_label:
        _ts.append(_ax.text(_x, _y, _lbl, fontsize=8, alpha=0.9))
    adjust_text(_ts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=_ax)

    # Aesthetics
    _ax.set_yticks(_ys)
    _ax.set_yticklabels([f"{_name}\n(n={_count})" for _name, _count in zip(df_fencing_dist['League'], df_fencing_dist['count'])], fontsize=10)
    _ax.set_xlabel(f'{model_choice} Score', fontsize=12)
    _ax.grid(axis='x', linestyle='--', alpha=0.5)
    _ax.set_title(f"{model_choice} Score Distributions (Grouped by {group_col})", pad=20, fontsize=14, fontweight='bold')
    _ax.legend(loc='upper right')
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hierarchichal levels comparison
    """)
    return


@app.cell
def _(df_league_depths, group_col, model_choice, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, max(4, len(df_league_depths) * 0.5)))
    _df_sorted = df_league_depths.sort_values(by='Number of Levels', ascending=True)

    # Swap label and color dynamically depending on the active model
    _val_label = "tiers" if model_choice == "SpringRank" else "spread"
    _bar_color = '#2ca02c' if model_choice == "SpringRank" else '#9467bd'

    _bars = _ax.barh(_df_sorted['League'], _df_sorted['Number of Levels'], color=_bar_color, alpha=0.8, edgecolor='black')

    # Annotate each bar
    for _bar in _bars:
        _width = _bar.get_width()
        _ax.text(_width + 0.05, _bar.get_y() + _bar.get_height()/2, f"{_width:.2f} {_val_label}", 
                va='center', ha='left', fontsize=9, fontweight='bold')

    # Aesthetics
    _ax.set_xlabel(f'Competition Spread / Depth ({_val_label})', fontsize=11)
    _ax.set_title(f'Competition Depth Comparison ({model_choice}) by {group_col}', fontsize=13, fontweight='bold')
    _ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation
    """)
    return


@app.cell
def _(all_fencing_bouts, df_fencer_scores, plt):
    # Match SpringRank scores back to their average "Post-Pool Rank"
    df_ranks = all_fencing_bouts[['Fencer A', 'Post-Pool Rank A']].dropna().rename(
        columns={'Fencer A': 'Fencer', 'Post-Pool Rank A': 'Post_Pool_Rank'}
    )

    # Merge with your calculated SpringRank scores
    df_validation = df_fencer_scores.merge(df_ranks.groupby('Fencer')['Post_Pool_Rank'].mean().reset_index(), on='Fencer')

    # Plot the correlation!
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(df_validation['Calculated_Score'], df_validation['Post_Pool_Rank'], alpha=0.5, color='purple')
    _ax.set_xlabel('Calculated Score')
    _ax.set_ylabel('Avg Post-Pool Rank')
    _ax.set_title('Validation: Calculated Score vs Official Post-Pool Ranks')
    _ax.invert_yaxis() # Invert because rank 1 is the best!
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    probably ignore this chart. When I have more time I'll try to match scores to official FIE rankings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Correlation with actual fie rankings
    """)
    return


@app.cell
def _(df_fencer_scores, model_choice):
    def plot_score_vs_fie_ranking(df_scores, rankings_dir="rankings"):
        import os
        import unicodedata
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr, pearsonr

        # Support files: ME, MF, MS, WE, WF, WS
        codes = ["ME", "MF", "MS", "WE", "WF", "WS"]
        fie_dfs = []

        for code in codes:
            path = os.path.join(rankings_dir, f"{code}-detailed-ranking-2026.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if "Name" in df.columns and "Rank" in df.columns:
                        df = df[["Rank", "Name"]].dropna()
                        df["Code"] = code
                        fie_dfs.append(df)
                except Exception:
                    continue

        if not fie_dfs:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No FIE ranking CSV files found in 'rankings/' directory.", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            return plt.gca()

        df_fie = pd.concat(fie_dfs, ignore_index=True)

        # Clean name helper to strip accents, lowercase, and sort words
        def get_match_key(name):
            if not isinstance(name, str):
                return ""
            # Normalize accents
            normalized = unicodedata.normalize('NFD', name)
            stripped = "".join(c for c in normalized if unicodedata.category(c) != 'Mn')
            # Keep alphanumeric characters and lowercase
            clean = "".join(c for c in stripped.lower() if c.isalnum() or c.isspace())
            # Sort words alphabetically to handle "First Last" vs "Last First" mismatch
            words = sorted(clean.split())
            return " ".join(words)

        df_fie["match_key"] = df_fie["Name"].apply(get_match_key)

        df_scores_copy = df_scores.copy()
        df_scores_copy["match_key"] = df_scores_copy["Fencer"].apply(get_match_key)

        # Merge SpringRank calculated scores with official FIE World Rankings
        merged = pd.merge(df_scores_copy, df_fie, on="match_key", how="inner")

        if len(merged) < 3:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"Too few matches found between datasets (Only {len(merged)} matches).\nCheck fencer name formats.", 
                    ha='center', va='center', fontsize=11, color='orange')
            ax.axis('off')
            return plt.gca()

        # Create visualization
        fig, ax = plt.subplots(figsize=(9, 6.5))

        # Calculate overall Spearman rank correlation
        spearman_corr, p_val = spearmanr(merged["Calculated_Score"], merged["Rank"])
        pearson_corr, _ = pearsonr(merged["Calculated_Score"], merged["Rank"])

        unique_codes = sorted(merged["Code"].unique())
        colors_list = plt.cm.tab10.colors

        for idx, code in enumerate(unique_codes):
            sub = merged[merged["Code"] == code]
            ax.scatter(
                sub["Calculated_Score"], 
                sub["Rank"], 
                label=f"{code} (n={len(sub)})", 
                alpha=0.75, 
                s=60,
                edgecolors='black',
                linewidths=0.5,
                color=colors_list[idx % len(colors_list)]
            )

        # Add general trendline
        if len(merged) > 1:
            m, b = np.polyfit(merged["Calculated_Score"], merged["Rank"], 1)
            x_domain = np.linspace(merged["Calculated_Score"].min(), merged["Calculated_Score"].max(), 100)
            ax.plot(x_domain, m * x_domain + b, color="red", linestyle="--", linewidth=1.5, label="Overall Trend")

        ax.set_xlabel(f"Calculated Score ({model_choice})", fontsize=11, fontweight='bold')
        ax.set_ylabel("Official FIE World Rank", fontsize=11, fontweight='bold')
        ax.set_title(
            f"Validation: Calculated Scores vs. Official FIE World Rankings\n"
            f"Spearman's ρ: {spearman_corr:.3f} (p = {p_val:.2e}) | Pearson's r: {pearson_corr:.3f}", 
            fontsize=12, 
            fontweight='bold', 
            pad=15
        )

        ax.invert_yaxis()  # Rank 1 is the best and should be at the top
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')

        plt.tight_layout()
        return plt.gca()

    plot_score_vs_fie_ranking(df_fencer_scores)
    return


@app.cell
def _(df_fencer_scores, os, pd):
    # Cell 1: Clean and Merge SpringRank Scores with FIE Rankings
    import unicodedata

    def clean_fencer_name(name):
        if not isinstance(name, str):
            return ""
        # Strip accents, lowercase, and keep alphanumeric characters
        normalized = unicodedata.normalize('NFD', name)
        stripped = "".join(c for c in normalized if unicodedata.category(c) != 'Mn')
        clean_str = "".join(c for c in stripped.lower() if c.isalnum() or c.isspace())
        # Sort words alphabetically to handle "Last First" vs "First Last" issues
        return " ".join(sorted(clean_str.split()))

    # 1. Load FIE CSVs
    fie_categories = ["ME", "MF", "MS", "WE", "WF", "WS"]
    fie_data_list = []

    for cat_code in fie_categories:
        csv_file_path = f"rankings/{cat_code}-detailed-ranking-2026.csv"
        if os.path.exists(csv_file_path):
            try:
                temp_df = pd.read_csv(csv_file_path)
                if "Name" in temp_df.columns and "Rank" in temp_df.columns:
                    temp_df = temp_df[["Rank", "Name", "Nat."]].dropna()
                    temp_df["Category"] = cat_code
                    fie_data_list.append(temp_df)
            except Exception as e:
                continue

    if fie_data_list:
        df_all_fie = pd.concat(fie_data_list, ignore_index=True)
        df_all_fie["match_key"] = df_all_fie["Name"].apply(clean_fencer_name)

        # 2. Normalize names in our SpringRank scores
        df_scores_normalized = df_fencer_scores.copy()
        df_scores_normalized["match_key"] = df_scores_normalized["Fencer"].apply(clean_fencer_name)

        # 3. Merge
        df_merged_rankings = pd.merge(
            df_scores_normalized, 
            df_all_fie, 
            on="match_key", 
            how="inner"
        )
    else:
        df_merged_rankings = pd.DataFrame()
    return df_merged_rankings, unicodedata


@app.cell
def _(df_merged_rankings, mo, pd, pearsonr, spearmanr):
    # Cell 2: Calculate and Display Correlations
    if len(df_merged_rankings) >= 3:
        correlation_results = []

        # Calculate global correlation
        glob_spearman, glob_s_p = spearmanr(df_merged_rankings["Calculated_Score"], df_merged_rankings["Rank"])
        glob_pearson, glob_p_p = pearsonr(df_merged_rankings["Calculated_Score"], df_merged_rankings["Rank"])

        correlation_results.append({
            "Category": "GLOBAL (All Overlapping Fencers)",
            "Fencers Matched": len(df_merged_rankings),
            "Spearman (Rank Corr)": glob_spearman,
            "Spearman p-value": glob_s_p,
            "Pearson (Linear Corr)": glob_pearson,
            "Pearson p-value": glob_p_p
        })

        # Calculate category-wise correlation
        for category, sub_grp in df_merged_rankings.groupby("Category"):
            if len(sub_grp) >= 3:
                s_corr, s_p = spearmanr(sub_grp["Calculated_Score"], sub_grp["Rank"])
                p_corr, p_p = pearsonr(sub_grp["Calculated_Score"], sub_grp["Rank"])
                correlation_results.append({
                    "Category": category,
                    "Fencers Matched": len(sub_grp),
                    "Spearman (Rank Corr)": s_corr,
                    "Spearman p-value": s_p,
                    "Pearson (Linear Corr)": p_corr,
                    "Pearson p-value": p_p
                })

        df_correlations = pd.DataFrame(correlation_results)

        correlation_display = mo.vstack([
            mo.md("### 📈 Relationship: SpringRank Scores vs. Official FIE World Rankings"),
            mo.md(
                "Below is the statistical correlation between your custom-calculated SpringRank scores "
                "and the official FIE world ranks. \n\n"
                "* **Spearman (Rank Correlation):** Measures monotonic relationship (how well order is preserved). A value close to **-1.0** is expected since Rank 1 (best) should correlate with the highest SpringRank Score.\n"
                "* **Pearson (Linear Correlation):** Measures linear relationship. Expect a value close to **-1.0**."
            ),
            mo.ui.table(df_correlations)
        ])
    else:
        correlation_display = mo.md("⚠️ **No overlapping fencers found.** Please ensure your FIE rankings CSV files are populated in the `rankings/` directory and names match your bout files.")

    correlation_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### clean correlations
    """)
    return


@app.cell
def _(mo):
    # Define the rank limit slider to let the user widen the validation filter
    rank_limit_slider = mo.ui.slider(start=50, stop=2000, step=50, value=200, label="Max FIE Rank Filter")
    mo.vstack([
        mo.md("### ⚙️ Adjust Validation Strictness"),
        mo.md("If your tournament dataset contains fewer top elite athletes, widen this slider to capture unranked/lower-ranked fencers:"),
        rank_limit_slider
    ])
    return (rank_limit_slider,)


@app.cell
def _(
    df_fencer_scores,
    download_status_message,
    mo,
    np,
    os,
    pd,
    rank_limit_slider,
    st,
    unicodedata,
):
    # Robust Self-Contained Alignment & Calculations (No re-imports to avoid cycles)
    _ = download_status_message # 👈 FORCE Marimo to re-run this cell whenever the downloader cell runs!

    _loaded_fie_dfs = {}
    _missing_files = []
    _corrupted_files = []

    # Local helper functions defined inside this cell scope to prevent global namespace collisions
    def _local_clean_name(name):
        if not isinstance(name, str):
            return ""
        # unicodedata is already globally imported upstream, so we just use it directly
        _norm = unicodedata.normalize('NFD', name)
        _stripped = "".join(_char for _char in _norm if unicodedata.category(_char) != 'Mn')
        _clean_str = "".join(_char for _char in _stripped.lower() if _char.isalnum() or _char.isspace())
        return " ".join(sorted(_clean_str.split()))

    def _local_get_fie_code(league_name):
        _name_lower = str(league_name).lower()
        _g = "W" if " / f" in _name_lower or "women" in _name_lower or "ladies" in _name_lower else "M" if " / m" in _name_lower or "men" in _name_lower or "mens" in _name_lower else None
        _w = "E" if "epee" in _name_lower or "épée" in _name_lower else "F" if "foil" in _name_lower else "S" if "sabre" in _name_lower else None
        return f"{_g}{_w}" if _g and _w else None

    # Load FIE CSVs directly using globally imported 'os' and 'pd'
    _fie_categories = ["ME", "MF", "MS", "WE", "WF", "WS"]

    for _cat in _fie_categories:
        _path = f"rankings/{_cat}-detailed-ranking-2026.csv"
        if os.path.exists(_path):
            try:
                _df = pd.read_csv(_path)
                _df.columns = [str(_col).strip() for _col in _df.columns]
            
                _rank_col = next((_c for _c in _df.columns if _c.lower() == "rank"), None)
                _name_col = next((_c for _c in _df.columns if _c.lower() == "name"), None)
                _nat_col = next((_c for _c in _df.columns if "nat" in _c.lower()), None)
                _pts_col = next((_c for _c in _df.columns if "point" in _c.lower() or "pts" in _c.lower()), None)
            
                if _rank_col and _name_col:
                    _cols_to_keep = [_rank_col, _name_col]
                    if _nat_col:
                        _cols_to_keep.append(_nat_col)
                    if _pts_col:
                        _cols_to_keep.append(_pts_col)
                    
                    _cleaned_df = _df[_cols_to_keep].dropna(subset=[_rank_col, _name_col]).copy()
                
                    _rename_map = {_rank_col: "Rank", _name_col: "Name"}
                    if _nat_col:
                        _rename_map[_nat_col] = "Nat."
                    if _pts_col:
                        _rename_map[_pts_col] = "FIE_Points"
                    _cleaned_df = _cleaned_df.rename(columns=_rename_map)
                
                    _cleaned_df["match_key"] = _cleaned_df["Name"].apply(_local_clean_name)
                    _loaded_fie_dfs[_cat] = (_cleaned_df, "FIE_Points" if _pts_col else None)
                else:
                    _corrupted_files.append((_cat, f"Missing columns. Columns found: {_df.columns.tolist()}"))
            except Exception as _e:
                _corrupted_files.append((_cat, str(_e)))
        else:
            _missing_files.append(_path)

    _diagnostics = {
        "fencer_scores_exists": 'df_fencer_scores' in globals() and not df_fencer_scores.empty,
        "fie_files_loaded": len(_loaded_fie_dfs) > 0,
        "sample_calculated_names": []
    }

    if _diagnostics["fencer_scores_exists"]:
        _diagnostics["sample_calculated_names"] = list(df_fencer_scores["Fencer"].dropna().unique()[:5])

    # Align Scores
    _matched_records = []
    if _diagnostics["fencer_scores_exists"] and _diagnostics["fie_files_loaded"]:
        for _, _row in df_fencer_scores.iterrows():
            _league = _row["League"]
            _cat_code = _local_get_fie_code(_league)
        
            if _cat_code in _loaded_fie_dfs:
                _fie_df, _pts_col_name = _loaded_fie_dfs[_cat_code]
                _fencer_clean = _local_clean_name(_row["Fencer"])
            
                _match = _fie_df[_fie_df["match_key"] == _fencer_clean]
                if not _match.empty:
                    _rec = {
                        "Fencer": _row["Fencer"],
                        "League": _league,
                        "Category": _cat_code,
                        "Calculated_Score": _row["Calculated_Score"],
                        "Official_Rank": _match.iloc[0]["Rank"],
                        "Country": _match.iloc[0]["Nat."] if "Nat." in _match.columns else "N/A"
                    }
                    if _pts_col_name and "FIE_Points" in _match.columns:
                        _rec["FIE_Points"] = pd.to_numeric(_match.iloc[0]["FIE_Points"], errors="coerce")
                    _matched_records.append(_rec)

    df_matched_validation = pd.DataFrame(_matched_records)

    # Filter and Compute Correlations Reactively
    _rank_limit = rank_limit_slider.value
    if not df_matched_validation.empty:
        df_elite = df_matched_validation[df_matched_validation["Official_Rank"] <= _rank_limit].copy()
    else:
        df_elite = pd.DataFrame()

    _correlations = []

    if not df_elite.empty and len(df_elite) >= 3:
        # Global Elite Correlations (using globally imported 'st' for scipy.stats)
        _glob_s_rank, _glob_s_rank_p = st.spearmanr(df_elite["Calculated_Score"], df_elite["Official_Rank"])
        _glob_k_rank, _glob_k_rank_p = st.kendalltau(df_elite["Calculated_Score"], df_elite["Official_Rank"])
    
        _res_glob = {
            "League / Category": f"GLOBAL Elite (Top {_rank_limit})",
            "Elite Fencers": len(df_elite),
            "Spearman ρ (vs Rank)": _glob_s_rank,
            "Spearman p-value": _glob_s_rank_p,
            "Kendall τ (vs Rank)": _glob_k_rank,
            "Kendall p-value": _glob_k_rank_p,
        }
    
        if "FIE_Points" in df_elite.columns and df_elite["FIE_Points"].notna().sum() >= 3:
            _sub_pts_glob = df_elite.dropna(subset=["FIE_Points"])
            _s_pts, _ = st.spearmanr(_sub_pts_glob["Calculated_Score"], _sub_pts_glob["FIE_Points"])
            _k_pts, _ = st.kendalltau(_sub_pts_glob["Calculated_Score"], _sub_pts_glob["FIE_Points"])
            _res_glob["Spearman ρ (vs Points)"] = _s_pts
            _res_glob["Kendall τ (vs Points)"] = _k_pts
        else:
            _res_glob["Spearman ρ (vs Points)"] = np.nan
            _res_glob["Kendall τ (vs Points)"] = np.nan
        
        _correlations.append(_res_glob)

        # League-specific correlations
        for _league_name, _sub_grp in df_elite.groupby("League"):
            if len(_sub_grp) >= 3:
                _s_rank, _s_rank_p = st.spearmanr(_sub_grp["Calculated_Score"], _sub_grp["Official_Rank"])
                _k_rank, _k_rank_p = st.kendalltau(_sub_grp["Calculated_Score"], _sub_grp["Official_Rank"])
            
                _res = {
                    "League / Category": _league_name,
                    "Elite Fencers": len(_sub_grp),
                    "Spearman ρ (vs Rank)": _s_rank,
                    "Spearman p-value": _s_rank_p,
                    "Kendall τ (vs Rank)": _k_rank,
                    "Kendall p-value": _k_rank_p,
                }
            
                if "FIE_Points" in _sub_grp.columns and _sub_grp["FIE_Points"].notna().sum() >= 3:
                    _sub_pts = _sub_grp.dropna(subset=["FIE_Points"])
                    _s_pts, _ = st.spearmanr(_sub_pts["Calculated_Score"], _sub_pts["FIE_Points"])
                    _k_pts, _ = st.kendalltau(_sub_pts["Calculated_Score"], _sub_pts["FIE_Points"])
                    _res["Spearman ρ (vs Points)"] = _s_pts
                    _res["Kendall τ (vs Points)"] = _k_pts
                else:
                    _res["Spearman ρ (vs Points)"] = np.nan
                    _res["Kendall τ (vs Points)"] = np.nan
                
                _correlations.append(_res)
            
        df_correlations_summary = pd.DataFrame(_correlations)
    
        _output_display = mo.vstack([
            mo.md(f"### 📊 Dynamic Validation Correlation (Top {_rank_limit})"),
            mo.ui.table(df_correlations_summary)
        ])
    else:
        # 4. Detailed Diagnostic Report
        _diagnostics_report = []
        _diagnostics_report.append("### 🔍 Validation Alignment Diagnostics")
    
        if not _diagnostics["fencer_scores_exists"]:
            _diagnostics_report.append("❌ **Error:** `df_fencer_scores` is empty. Please run your ranking model first.")
        else:
            _diagnostics_report.append("✅ `df_fencer_scores` is calculated.")
        
        if not _diagnostics["fie_files_loaded"]:
            _diagnostics_report.append("❌ **Error:** No FIE detailed ranking files could be loaded.")
            if _missing_files:
                _diagnostics_report.append(f"  * **Missing files:** `{_missing_files}`")
            if _corrupted_files:
                _diagnostics_report.append(f"  * **Corrupted or unparseable files:** `{_corrupted_files}`")
        else:
            _diagnostics_report.append(f"✅ Successfully loaded {len(_loaded_fie_dfs)} FIE ranking files on the fly.")
        
        if _diagnostics["fencer_scores_exists"] and _diagnostics["fie_files_loaded"]:
            if df_matched_validation.empty:
                _diagnostics_report.append("❌ **Error:** Name alignment failed completely (0 overlap matches).")
                _diagnostics_report.append(f"  * Sample calculated fencers in your dataset: `{_diagnostics['sample_calculated_names']}`")
                _first_loaded_cat = list(_loaded_fie_dfs.keys())[0]
                _sample_fie_names = list(_loaded_fie_dfs[_first_loaded_cat][0]["Name"].unique()[:5])
                _diagnostics_report.append(f"  * Sample fencers in loaded FIE files: `{_sample_fie_names}`")
                _diagnostics_report.append("⚠️ *Check spelling formats. Your cleaning function may need updating if formats are mismatched.*")
            else:
                _diagnostics_report.append(f"✅ Found **{len(df_matched_validation)}** overlapping matches overall.")
                _diagnostics_report.append(f"❌ However, **0** matched fencers had an official FIE Rank of $$\\le {_rank_limit}$$.")
                _diagnostics_report.append(f"  * Best matched rank in your dataset: **FIE Rank {df_matched_validation['Official_Rank'].min()}**")
                _diagnostics_report.append("👉 **Use the slider above to widen the FIE Rank filter (e.g., to 500 or 1000) to capture your athletes!**")
            
        _output_display = mo.vstack([
            mo.md("\n".join(_diagnostics_report))
        ])

    _output_display
    return df_elite, df_matched_validation


@app.cell
def _(df_elite, mo, model_choice, pd, plt, sns):
    # Binned Rank Boxplot for Top 200 (Proves the strong Monotonic Trend)

    if 'df_elite' in globals() and not df_elite.empty:
        # Create rank bins specifically for the top 200
        df_elite['Rank_Bin'] = pd.cut(
            df_elite['Official_Rank'], 
            bins=[0, 50, 100, 150, 200], 
            labels=['1-50', '51-100', '101-150', '151-200']
        )
    
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_elite, 
            x='Rank_Bin', 
            y='Calculated_Score', 
            hue='Rank_Bin',  # Fixes the deprecated warning
            palette='viridis', 
            legend=False
        )
        plt.title('Elite Top 200: SpringRank Score Distribution by FIE Rank Bin', fontweight='bold')
        plt.xlabel('Official FIE Rank Bin', fontweight='bold')
        plt.ylabel(f'Calculated Score ({model_choice})', fontweight='bold')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.show()

    else:
        mo.md("Run the alignment cell first to generate df_elite.")
    return


@app.cell
def _(df_elite, pd, plt, sns):
    # Create rank bins
    df_elite['Rank_Bin'] = pd.cut(df_elite['Official_Rank'], bins=[0, 100, 500, 1000], labels=['1-100', '101-500', '501-1000'])

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_elite, x='Rank_Bin', y='Calculated_Score', palette='viridis')
    plt.title('Distribution of SpringRank Scores across FIE Rank Bins')
    plt.xlabel('Official FIE Rank Bin')
    plt.ylabel('Calculated Score')
    plt.show()
    return


@app.cell
def _(df_elite, mo, model_choice, pd, plt, sns):
    # Cell 45: Boxplot Distribution across Ranks
    if 'df_elite' in globals() and not df_elite.empty:
        # Copy the dataframe so we don't modify the global df_elite permanently
        _df_plot = df_elite.copy()
    
        # Dynamically bin FIE ranks into logical categories (adjusts automatically based on your slider)
        _max_rank = _df_plot['Official_Rank'].max()
        _bins = [0, 50, 100, max(200, _max_rank)]
        _labels = ['1-50', '51-100', f'101-{int(_max_rank)}']
    
        _df_plot['Rank_Bin'] = pd.cut(
            _df_plot['Official_Rank'], 
            bins=_bins, 
            labels=_labels
        )
    
        # Drop fencers outside the binned range
        _df_plot = _df_plot.dropna(subset=['Rank_Bin'])
    
        if not _df_plot.empty:
            _fig, _ax = plt.subplots(figsize=(8, 5))
        
            # Draw the boxplot safely
            sns.boxplot(
                data=_df_plot, 
                x='Rank_Bin', 
                y='Calculated_Score', 
                hue='Rank_Bin',
                palette='viridis',
                legend=False,
                ax=_ax
            )
        
            # Aesthetics
            _ax.set_title('Distribution of Calculated Scores across FIE Rank Bins', fontsize=12, fontweight='bold', pad=15)
            _ax.set_xlabel('Official FIE Rank Bin', fontweight='bold')
            _ax.set_ylabel(f'Calculated Score ({model_choice})', fontweight='bold')
            _ax.grid(axis='y', linestyle=':', alpha=0.5)
        
            plt.tight_layout()
            _box_viz = plt.gca()
        else:
            _box_viz = mo.md("No fencers fit into the rank bins. Try widening your FIE rank slider.")
    else:
        _box_viz = mo.md("Calculate elite scores first to view distribution.")

    _box_viz
    return


@app.cell
def _(df_fencer_scores, os):
    # 1. Check if the FIE files exist on disk

    for _cat in ["ME", "MF", "MS", "WE", "WF", "WS"]:
        _path = f"rankings/{_cat}-detailed-ranking-2026.csv"
        print(f"{_cat} exists: {os.path.exists(_path)}")

    # 2. Check what leagues are in your source data
    if 'df_fencer_scores' in globals():
        print("\nLeague names in your source data:")
        print(df_fencer_scores["League"].unique())

    # 3. Test the mapping function against your specific names
    def _get_fie_code(league_name):
        _name_lower = str(league_name).lower()
        _g = "M" if " / m" in _name_lower or "men" in _name_lower else "W" if " / f" in _name_lower or "women" in _name_lower else None
        _w = "E" if "epee" in _name_lower or "épée" in _name_lower else "F" if "foil" in _name_lower else "S" if "sabre" in _name_lower else None
        return f"{_g}{_w}" if _g and _w else None

    print("\nTest mapping:")
    for _league in df_fencer_scores["League"].unique():
        print(f"{_league} -> {_get_fie_code(_league)}")
    return


@app.cell
def _(colors, df_matched_validation, mo, model_choice, np, plt):
    # Dynamic Visual Validation (Ranks and Points side-by-side)
    if 'df_matched_validation' in globals() and not df_matched_validation.empty and len(df_matched_validation) >= 3:
        _has_points = "FIE_Points" in df_matched_validation.columns and df_matched_validation["FIE_Points"].notna().sum() > 2
    
        _fig, _axs = plt.subplots(1, 2 if _has_points else 1, figsize=(14, 6) if _has_points else (8, 6))
        _ax_rank = _axs[0] if _has_points else _axs
    
        _unique_categories = sorted(df_matched_validation["Category"].unique())
    
        # Plot 1: Ranks
        for _idx, _cat in enumerate(_unique_categories):
            _sub = df_matched_validation[df_matched_validation["Category"] == _cat]
            _color = colors[_idx % len(colors)]
        
            _ax_rank.scatter(
                _sub["Calculated_Score"], _sub["Official_Rank"], 
                label=f"{_cat} (n={len(_sub)})", alpha=0.75, s=60, edgecolors='black', linewidths=0.5, color=_color
            )
            if len(_sub) > 2:
                _m, _b = np.polyfit(_sub["Calculated_Score"], _sub["Official_Rank"], 1)
                _x_dom = np.linspace(_sub["Calculated_Score"].min(), _sub["Calculated_Score"].max(), 100)
                _ax_rank.plot(_x_dom, _m * _x_dom + _b, color=_color, linestyle="--", alpha=0.5)
            
        _ax_rank.set_xlabel(f"Calculated Score ({model_choice})", fontweight='bold')
        _ax_rank.set_ylabel("Official FIE World Rank", fontweight='bold')
        _ax_rank.set_title("Validation vs. Official Rank (Inverted)", fontweight='bold')
        _ax_rank.invert_yaxis()
        _ax_rank.grid(True, linestyle=":", alpha=0.6)
        _ax_rank.legend()
    
        # Plot 2: Points (If available)
        if _has_points:
            _ax_pts = _axs[1]
            for _idx, _cat in enumerate(_unique_categories):
                _sub = df_matched_validation[(df_matched_validation["Category"] == _cat) & (df_matched_validation["FIE_Points"].notna())]
                _color = colors[_idx % len(colors)]
            
                _ax_pts.scatter(
                    _sub["Calculated_Score"], _sub["FIE_Points"], 
                    label=f"{_cat} (n={len(_sub)})", alpha=0.75, s=60, edgecolors='black', linewidths=0.5, color=_color
                )
                if len(_sub) > 2:
                    _m, _b = np.polyfit(_sub["Calculated_Score"], _sub["FIE_Points"], 1)
                    _x_dom = np.linspace(_sub["Calculated_Score"].min(), _sub["Calculated_Score"].max(), 100)
                    _ax_pts.plot(_x_dom, _m * _x_dom + _b, color=_color, linestyle="--", alpha=0.5)
                
            _ax_pts.set_xlabel(f"Calculated Score ({model_choice})", fontweight='bold')
            _ax_pts.set_ylabel("Official FIE World Points", fontweight='bold')
            _ax_pts.set_title("Validation vs. Total FIE Points", fontweight='bold')
            _ax_pts.grid(True, linestyle=":", alpha=0.6)
            _ax_pts.legend()

        plt.tight_layout()
        _viz_output = plt.gca()
    else:
        _viz_output = mo.md("Nothing to plot. Check loaded fencer dataset.")
    
    _viz_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Fantasy Fencing Matchups:
    A fun little application of the score calculations. Here you can select any two athletes in the dataset from any weapon at it will predict the likely outcome of a match between them. Not to be taken too seriously obviouly
    """)
    return


@app.cell
def _(df_fencer_scores, mo):
    # Get all valid fencer names from the ranking dataframe
    fencer_names = sorted(
        df_fencer_scores["Fencer"]
        .dropna()
        .unique()
        .tolist()
    )

    fencer_1 = mo.ui.dropdown(
        options=fencer_names,
        label="Fencer 1:",
        value=fencer_names[0]
    )

    fencer_2 = mo.ui.dropdown(
        options=fencer_names,
        label="Fencer 2:",
        value=fencer_names[1]
    )

    mo.vstack([
        fencer_1,
        fencer_2
    ])
    return fencer_1, fencer_2


@app.cell
def _(df_fencer_scores, df_league_depths, fencer_1, fencer_2, mo, np):
    # Get selected fencer names
    name_1 = fencer_1.value
    name_2 = fencer_2.value

    # Get their calculated scores
    score_1 = df_fencer_scores.loc[
        df_fencer_scores["Fencer"] == name_1,
        "Calculated_Score"
    ].iloc[0]

    score_2 = df_fencer_scores.loc[
        df_fencer_scores["Fencer"] == name_2,
        "Calculated_Score"
    ].iloc[0]

    # Average beta across leagues
    target_beta = df_league_depths["Beta (Predictability)"].mean()

    # Logistic probability
    prob_1_wins = 1 / (
        1 + np.exp(-target_beta * (score_1 - score_2))
    )

    prob_2_wins = 1 - prob_1_wins

    mo.md(f"""
    ## ⚔️ Simulation Matchup

    | | Fencer | Calculated Score |
    |---|---|---:|
    | 🥇 Fencer 1 | **{name_1}** | {score_1:.3f} |
    | 🥈 Fencer 2 | **{name_2}** | {score_2:.3f} |

    **β (Predictability):** {target_beta:.3f}

    ### Predicted outcome

    **{name_1}: {prob_1_wins * 100:.2f}%**

    **{name_2}: {prob_2_wins * 100:.2f}%**
    """)
    return


if __name__ == "__main__":
    app.run()
