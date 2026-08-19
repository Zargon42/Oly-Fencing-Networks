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

    return AffinityPropagation, LabelEncoder, adjust_text, st


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colormap = plt.cm.tab10
    colors = {i: colormap(i) for i in range(20)}
    return (plt,)


@app.cell
def _():
    import cv_tools as cvtl
    from statsbombpy import sb

    return


@app.cell
def _():
    outdir_fig = '../figures/'
    lecture_id = 5
    return


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
            value='None',
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
