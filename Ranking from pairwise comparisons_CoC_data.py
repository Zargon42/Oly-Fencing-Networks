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
    # L5: rankings from pairwise comparisons
    Here we explore how to extract hidden rankings from pairwise comparisons, e.g. games between teams in sport.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
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

    return BLACK, bt, prcs, sr, tl, viz


@app.cell
def _():
    import scipy.stats as st
    from sklearn.cluster import AffinityPropagation
    from matplotlib.lines import Line2D
    from adjustText import adjust_text
    from scipy.stats import pearsonr, spearmanr

    return AffinityPropagation, Line2D, adjust_text, pearsonr, spearmanr, st


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colormap = plt.cm.tab10
    colors = {i: colormap(i) for i in range(20)}
    return colormap, plt


@app.cell
def _():
    import cv_tools as cvtl
    from statsbombpy import sb

    return (sb,)


@app.cell
def _():
    outdir_fig = '../figures/'
    lecture_id = 5
    return lecture_id, outdir_fig


@app.cell
def _(np):
    seed = 10
    prng = np.random.RandomState(seed)
    return (prng,)


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

    **Source**: download a dataset from [StatsBomb open data](https://github.com/statsbomb/open-data/tree/master).

    We will use the python package [`statsbombpy`](https://github.com/statsbomb/statsbombpy) to process the raw data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by downloading matches from at least two different competitions, to be able to compare them
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Import raw data
    """)
    return


@app.cell
def _(sb):
    df_comp = sb.competitions()
    return (df_comp,)


@app.cell
def _(df_comp):
    df_comp.head()
    return


@app.cell
def _(df_comp):
    _mask = df_comp['competition_international'] == False
    df_comp_1 = df_comp[_mask]
    competitionId2Name = dict(zip(df_comp_1['competition_id'], df_comp_1['competition_name']))
    df_comp_1.competition_name.unique()
    return (competitionId2Name,)


@app.cell
def _():
    competition_ids = [37, 49, 12, 2, 11]
    season_ids = [90, 3, 27, 27, 27]
    compId2sort = {_c: i for i, _c in enumerate(competition_ids)}
    return compId2sort, competition_ids, season_ids


@app.cell
def _(competition_ids, sb, season_ids):
    games = {_c: sb.matches(competition_id=_c, season_id=season_ids[i]) for i, _c in enumerate(competition_ids)}
    return (games,)


@app.cell
def _(games):
    cols = ['match_id', 'match_date','home_team', 'away_team', 'home_score', 'away_score']
    games[49][cols].head()
    return (cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Process into a matrix
    """)
    return


@app.cell
def _(competition_ids, games, prcs):
    df = {competition_id: prcs.process_games(games[competition_id]) for competition_id in competition_ids}
    A, encoder_teams = ({}, {})
    for k, _v in df.items():
        A[k], encoder_teams[k] = prcs.df2matrix(_v, score_label='points', method='points')
        print(k, A[k].shape)
    return A, df, encoder_teams


@app.cell
def _(np, pd):
    def get_points(df: pd.DataFrame, competition_id: int=None):
        """
        Get total number of points for each team
        """
        df_home = df.groupby(by=['home_team'])['home_points'].agg(['count', 'sum']).reset_index()
        df_home.rename(columns={'home_team': 'node_label', 'count': 'n_matches', 'sum': 'points'}, inplace=True)
        df_away = df.groupby(by=['away_team'])['away_points'].agg(['count', 'sum']).reset_index()
        df_away.rename(columns={'away_team': 'node_label', 'count': 'n_matches', 'sum': 'points'}, inplace=True)
        df_points = pd.concat([df_home, df_away]).reset_index().groupby(by=['node_label'])[['points', 'n_matches']].agg(['sum']).droplevel(1, axis=1).reset_index()
        df_points.loc[:, 'points_prg'] = (df_points['points'] / df_points['n_matches']).map(lambda _x: np.round(_x, 2))
        df_points = df_points.sort_values(by='points_prg', ascending=False).reset_index(drop=True)
        if competition_id is not None:
            df_points.loc[:, 'competition_id'] = competition_id
        return df_points

    return (get_points,)


@app.cell
def _(df, get_points, pd):
    df_points_comp = {k: get_points(_v, competition_id=k) for k, _v in df.items()}
    df_points = pd.concat(df_points_comp.values())
    return df_points, df_points_comp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # import fencing data
    """)
    return


@app.cell
def _(pd):
    bouts_pool = pd.read_csv("CoC dataset/Pools.csv")
    bouts_DE = pd.read_csv("CoC dataset/DEs.csv")
    athletes = pd.read_csv("CoC dataset/Athletes.csv")
    return bouts_DE, bouts_pool


@app.cell
def _(bouts_pool):
    bouts_pool.head()
    return


@app.cell
def _(pd):
    competitions = pd.read_csv("Womens foil olympics/all_womens_foil_tournament_data_May_13_2021_cleaned.csv")
    competitions.head(50)
    return


@app.cell
def _(bouts_DE, bouts_pool, np, pd):
    from sklearn.preprocessing import LabelEncoder

    # Clean data and ensure no NaN values in crucial columns
    _bouts_pool_clean = bouts_pool.dropna(subset=['Fencer A', 'Fencer B', 'Event Name', 'Season']).copy()
    _bouts_DE_clean = bouts_DE.dropna(subset=['Fencer A', 'Fencer B', 'Event Name', 'Season']).copy()

    # Create a unique key for each competition (Event Name + Season)
    _bouts_pool_clean['competition_key'] = _bouts_pool_clean['Event Name'] + ' (' + _bouts_pool_clean['Season'].astype(str) + ')'
    _bouts_DE_clean['competition_key'] = _bouts_DE_clean['Event Name'] + ' (' + _bouts_DE_clean['Season'].astype(str) + ')'

    # Combine to count total bouts per competition key to find the most active ones
    _all_bouts = pd.concat([_bouts_pool_clean, _bouts_DE_clean], ignore_index=True)
    _comp_counts = _all_bouts['competition_key'].value_counts()

    # We select the top 5 competitions with the most bouts for stable analysis
    _top_competitions = _comp_counts.index[:5]

    fencing_A = {}
    fencing_encoders = {}
    fencing_competition_names = {}

    for _i, _comp_key in enumerate(_top_competitions):
        _df_comp = _all_bouts[_all_bouts['competition_key'] == _comp_key]
    
        # Fit LabelEncoder on all unique fencers in this competition
        _fencers = pd.concat([_df_comp['Fencer A'], _df_comp['Fencer B']]).unique()
        _le = LabelEncoder()
        _le.fit(_fencers)
        _n_fencers = len(_le.classes_)
    
        # Initialize adjacency matrix
        _A_mat = np.zeros((_n_fencers, _n_fencers))
    
        # Populate the matrix based on wins (A_ij = wins of i over j)
        for _, _row in _df_comp.iterrows():
            try:
                _u = _le.transform([_row['Fencer A']])[0]
                _v = _le.transform([_row['Fencer B']])[0]
            except ValueError:
                continue
            
            _winner = _row['Winner']
            if _winner == _row['Fencer A']:
                _A_mat[_u, _v] += 1
            elif _winner == _row['Fencer B']:
                _A_mat[_v, _u] += 1
            
        fencing_A[_i] = _A_mat
        fencing_encoders[_i] = _le
        fencing_competition_names[_i] = _comp_key

    # Extract node mappings for the fencing datasets
    nodeLabel2Id_fencing = {k: {_c: i for i, _c in enumerate(_v.classes_)} for k, _v in fencing_encoders.items()}
    nodeId2Label_fencing = {k: {i: _c for i, _c in enumerate(_v.classes_)} for k, _v in fencing_encoders.items()}
    return (LabelEncoder,)


@app.cell
def _(bouts_DE, bouts_pool, pd):
    # Combine pool and DE bouts
    all_fencing_bouts = pd.concat([bouts_pool, bouts_DE], ignore_index=True)

    # Drop rows with missing values in critical columns
    all_fencing_bouts = all_fencing_bouts.dropna(subset=['Fencer A', 'Fencer B', 'Winner']).copy()
    return (all_fencing_bouts,)


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

    # Dropdown allowing you to select which category defines a "league" with an optional second category to subdivide
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
    group_by_selector
    return (group_by_selector,)


@app.cell
def _(mo):
    # Create interactive controls to tune the rankings
    min_bouts_slider = mo.ui.slider(start=1, stop=20, step=1, value=5, label="Min Bouts Required")
    use_connectivity_filter = mo.ui.checkbox(label="Filter Isolated Events (Keep Largest Component)", value=True)

    mo.hstack([min_bouts_slider, use_connectivity_filter], justify="start")
    return min_bouts_slider, use_connectivity_filter


@app.cell
def computationcell(
    LabelEncoder,
    all_fencing_bouts,
    group_by_selector,
    min_bouts_slider,
    np,
    nx,
    pd,
    sr,
    use_connectivity_filter,
):
    # Declare reactive inputs
    group_col = group_by_selector.value
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
            # Skip groups that are too small to build a network
            if len(group_df) < 50:
                continue
            
            # 1. Count appearances per fencer in this group
            all_fencers_in_group = pd.concat([group_df['Fencer A'], group_df['Fencer B']])
            fencer_appearance_counts = all_fencers_in_group.value_counts()
        
            # Keep only fencers who meet the minimum bout threshold
            active_fencers = fencer_appearance_counts[fencer_appearance_counts >= min_bouts].index
        
            # Filter the bouts so that BOTH fencers meet the minimum active threshold
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
                # If the filter is off, keep all active fencers
                target_fencers = list(active_fencers)
            
            if len(target_fencers) < 5:
                continue
            
            # Filter dataframe to match final fencer selection
            filtered_df = filtered_df[
                filtered_df['Fencer A'].isin(target_fencers) & 
                filtered_df['Fencer B'].isin(target_fencers)
            ]
        
            # Re-initialize LabelEncoder with only target fencers
            le = LabelEncoder()
            le.fit(target_fencers)
            n_fencers = len(le.classes_)
        
            # Map fencer names to matrix indices
            fencer_to_id = {name: i for i, name in enumerate(le.classes_)}
            u_arr = filtered_df['Fencer A'].map(fencer_to_id).values.astype(int)
            v_arr = filtered_df['Fencer B'].map(fencer_to_id).values.astype(int)
            win_a_mask = (filtered_df['Winner'] == filtered_df['Fencer A']).values
        
            # 3. Initialize and build the adjacency matrix
            A_mat = np.zeros((n_fencers, n_fencers))
            np.add.at(A_mat, (u_arr[win_a_mask], v_arr[win_a_mask]), 1)
            np.add.at(A_mat, (v_arr[~win_a_mask], u_arr[~win_a_mask]), 1)
                
            # 4. Fit SpringRank model
            sr_model = sr.SpringRank()
            try:
                sr_model.fit(A_mat)
                ranks = sr_model.ranks
                beta = sr_model.get_beta()
                depth = sr_model.depth
                n_levels = sr_model.n_levels
            except Exception as e:
                continue
        
            # Store individual fencer scores
            for idx, fencer_name in enumerate(le.classes_):
                fencer_scores_list.append({
                    'Fencer': fencer_name,
                    'League': group_name,
                    'SpringRank_Score': ranks[idx]
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
    return df_fencer_scores, df_league_depths, group_col


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
            df_fencer_scores.sort_values(by='SpringRank_Score', ascending=False),
            label="Individual Fencer Rankings"
        )
    ])
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
    np,
    plt,
    st,
):
    # Create summary statistics for the fencing leagues
    df_fencing_dist = df_fencer_scores.groupby('League')['SpringRank_Score'].agg(['min', 'max', 'count', 'mean']).reset_index()

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
        _scores = _league_fencers['SpringRank_Score'].values
    
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
        _sorted_fencers = _league_fencers.sort_values(by='SpringRank_Score', ascending=False)
        if len(_sorted_fencers) > 0:
            # Top 2 fencers
            for _, _row in _sorted_fencers.head(2).iterrows():
                _fencers_to_label.append((_row['SpringRank_Score'], _y_val + np.random.normal(0, 0.04), _row['Fencer']))
            # Bottom 1 fencer
            if len(_sorted_fencers) > 2:
                _row_bot = _sorted_fencers.iloc[-1]
                _fencers_to_label.append((_row_bot['SpringRank_Score'], _y_val + np.random.normal(0, 0.04), _row_bot['Fencer']))

    # Add text labels and resolve overlap
    _ts = []
    for _x, _y, _lbl in _fencers_to_label:
        _ts.append(_ax.text(_x, _y, _lbl, fontsize=8, alpha=0.9))
    adjust_text(_ts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=_ax)

    # Aesthetics
    _ax.set_yticks(_ys)
    _ax.set_yticklabels([f"{_name}\n(n={_count})" for _name, _count in zip(df_fencing_dist['League'], df_fencing_dist['count'])], fontsize=10)
    _ax.set_xlabel('SpringRank Score', fontsize=12)
    _ax.grid(axis='x', linestyle='--', alpha=0.5)
    _ax.set_title(f"SpringRank Score Distributions (Grouped by {group_col})", pad=20, fontsize=14, fontweight='bold')
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
def _(df_league_depths, group_col, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, max(4, len(df_league_depths) * 0.5)))
    _df_sorted = df_league_depths.sort_values(by='Number of Levels', ascending=True)

    # Plot a clean horizontal bar chart
    _bars = _ax.barh(_df_sorted['League'], _df_sorted['Number of Levels'], color='#2ca02c', alpha=0.8, edgecolor='black')

    # Annotate each bar with the exact tier height
    for _bar in _bars:
        _width = _bar.get_width()
        _ax.text(_width + 0.05, _bar.get_y() + _bar.get_height()/2, f"{_width:.2f} tiers", 
                va='center', ha='left', fontsize=9, fontweight='bold')

    # Aesthetics
    _ax.set_xlabel('Number of Levels (Competition Depth)', fontsize=11)
    _ax.set_title(f'Distinct Hierarchical Levels by {group_col}', fontsize=13, fontweight='bold')
    _ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Run ranking models

    We can proceed by learning scores from the outcomes of matches
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 SpringRank
    """)
    return


@app.cell
def _(A, competitionId2Name, pd, sr):
    model = {}
    scaled_ranks = {}
    stats = []
    for k_1, _v in A.items():
        model[k_1] = sr.SpringRank()
        model[k_1].fit(_v)
        scaled_ranks[k_1] = model[k_1].get_rescaled_ranks(0.75)
        _d = [k_1, competitionId2Name[k_1], model[k_1].get_beta(), model[k_1].depth, model[k_1].n_levels, model[k_1].delta_beta]
        stats.append(_d)
    df_stats = pd.DataFrame(stats, columns=['competition_id', 'competition_name', 'beta', 'depth', 'n_levels', 'delta_level'])
    df_stats
    return df_stats, model, scaled_ranks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Bradley-Terry model
    """)
    return


@app.cell
def _(A, bt, np):
    model_bt = {}
    scaled_ranks_bt = {}
    for k_2, _v in A.items():
        model_bt[k_2] = bt.BradleyTerry()
        model_bt[k_2].fit(_v, method='em')
        scaled_ranks_bt[k_2] = np.exp(model_bt[k_2].ranks)
        scaled_ranks_bt[k_2] = model_bt[k_2].get_rescaled_ranks(0.75)
    return (scaled_ranks_bt,)


@app.cell
def _(encoder_teams):
    nodeLabel2Id = {k: {_c: i for i, _c in enumerate(_v.classes_)} for k, _v in encoder_teams.items()}
    nodeId2Label = {k: {i: _c for i, _c in enumerate(_v.classes_)} for k, _v in encoder_teams.items()}
    return nodeId2Label, nodeLabel2Id


@app.cell
def _(model, nodeId2Label, np, pd, scaled_ranks, scaled_ranks_bt):
    # df_res = pd.concat([pd.DataFrame({'node_id': np.arange(model[k].ranks.shape[0]),'node_label': [nodeId2Label[k][i] for i in np.arange(model[k].ranks.shape[0])], 'score': model[k].ranks, 'competition_id': [k for j in range(len(model[k].ranks))]})
    #            for k in model.keys()])
    show_rescaled = True
    fig_label = 'rescaled' if show_rescaled == True else 'not_rescaled'
    if show_rescaled == True:
        df_res = pd.concat([pd.DataFrame({'node_id': np.arange(_v.shape[0]), 'node_label': [nodeId2Label[k][i] for i in np.arange(_v.shape[0])], 'score_sr': _v, 'competition_id': [k for j in range(len(_v))]}) for k, _v in scaled_ranks.items()])
        df_res_bt = pd.concat([pd.DataFrame({'node_id': np.arange(_v.shape[0]), 'node_label': [nodeId2Label[k][i] for i in np.arange(_v.shape[0])], 'score_bt': _v, 'competition_id': [k for j in range(len(_v))]}) for k, _v in scaled_ranks_bt.items()])
        df_res = df_res.merge(df_res_bt, on=['node_id', 'node_label', 'competition_id'])
    else:
        df_res = pd.concat([pd.DataFrame({'node_id': np.arange(_v.ranks.shape[0]), 'node_label': [nodeId2Label[k][i] for i in np.arange(_v.ranks.shape[0])], 'score_sr': _v.ranks, 'competition_id': [k for j in range(len(_v.ranks))]}) for k, _v in model.items()])
        df_res_bt = pd.concat([pd.DataFrame({'node_id': np.arange(_v.ranks.shape[0]), 'node_label': [nodeId2Label[k][i] for i in np.arange(_v.ranks.shape[0])], 'score_bt': _v.ranks, 'competition_id': [k for j in range(len(_v.ranks))]}) for k, _v in model.items()])
    # df_res.head()
        df_res = df_res.merge(df_res_bt, on=['node_id', 'node_label', 'competition_id'])
    return df_res, fig_label


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's get aggregate statistics to characterize the distributions per league
    """)
    return


@app.cell
def _(compId2sort, df_res):
    algo = 'sr'
    metric = f'score_{algo}'
    df_plot_dist = df_res.groupby(by='competition_id')[metric].agg(['describe']).droplevel(0, axis=1).reset_index().sort_values(by='competition_id', key=lambda _x: _x.map(compId2sort))
    df_plot_dist
    return algo, df_plot_dist, metric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Analyze results
    """)
    return


@app.cell
def _(viz):
    mc = viz.default_colors_dict['blue_dark']
    ms = 200
    colors_1 = [viz.default_colors_dict['blue_sb_dark'], viz.default_colors_dict['green_forest'], viz.default_colors_dict['red_adobe'], viz.default_colors_dict['yellow_sand'], viz.default_colors_dict['purple'], viz.default_colors_dict['dark_grey'], viz.default_colors_dict['purple_sb_dark']]
    return colors_1, mc, ms


@app.cell
def _(competitionId2Name, df_plot_dist):
    sorted_ylabels = [competitionId2Name[_c] for _c in df_plot_dist['competition_id']]
    return (sorted_ylabels,)


@app.cell
def _():
    label_dict = {'sr':'SpringRank','bt':'Bradley-Terry','points_prg':'Points per game'}
    return (label_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot score distribution over different leagues
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    BLACK,
    Line2D,
    adjust_text,
    algo,
    colors_1,
    competition_ids,
    df_plot_dist,
    df_res,
    fig_label,
    label_dict,
    lecture_id,
    mc,
    metric,
    model,
    ms,
    np,
    outdir_fig,
    plt,
    sorted_ylabels,
    st,
    tl,
    viz,
):
    title = f'{label_dict[algo]} scores from soccer matches'
    point_label = 'node_label'
    nmax = min(200, len(df_plot_dist))
    n_display_max = 10
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))
    _xs = np.arange(len(df_plot_dist), 0, -1)
    plt.hlines(_xs[:nmax], xmin=df_plot_dist[:nmax]['min'], xmax=df_plot_dist[:nmax]['max'], alpha=0.7, color=mc, lw=2, ls='-', zorder=1)
    plt.scatter(df_plot_dist[:nmax]['max'], _xs[:nmax], s=ms, alpha=0.6, c=viz.default_colors_dict['blue_dark'], edgecolors=BLACK, zorder=5)
    plt.scatter(df_plot_dist[:nmax]['min'], _xs[:nmax], s=ms, alpha=0.6, c=viz.default_colors_dict['blue'], edgecolors=BLACK, zorder=5)
    '\nInidividual points\n'
    ylabels = []
    teams_to_display = []
    for i, cid in enumerate(competition_ids):
        _g = df_res[df_res.competition_id == cid]
        l = len(_g)
        x_data = np.array([_xs[i]] * l)
        x_jittered = np.array([_x + st.t(df=6, scale=0.08).rvs(1) for _x in x_data])
        xjit2name = dict(zip(_g[point_label], x_jittered))
        _x = np.array(_g[metric])
        clustering = AffinityPropagation(random_state=5).fit(_x.reshape(-1, 1))
        clabels = clustering.labels_
        n_clusters = len(np.unique(clabels))
        cs = [colors_1[k] for k in clabels]
        plt.scatter(_g[metric], x_jittered, s=50, alpha=0.8, c=cs, edgecolors=BLACK, zorder=1)
        _msg = f'{sorted_ylabels[i]} (n={l})'.replace("Women's", '')
        _msg = f'{_msg}\nbeta = {model[cid].get_beta():.2f}'
        _msg = _msg.replace('Women', '')
        ylabels.append(f'{_msg}')
        _cond1 = _g[metric] >= _g[metric].quantile(0.8)
        _cond2 = _g[metric] <= _g[metric].quantile(0.2)
        _mask = np.logical_or(_cond1, _cond2)
        n_display = min(n_display_max, np.sum(_mask))
        for i in range(n_display):
            _df_tmp = _g[_mask].sort_values(by=[metric], ascending=False)
            tname = _df_tmp.iloc[i][point_label]
            _y = _df_tmp.iloc[i][metric]
            _x = x_jittered[_mask][i]
            teams_to_display.append([_y, xjit2name[tname], tname])
    _ts = []
    for _d in teams_to_display:
        _msg = f'{_d[2]}'
        _ts.append(_ax.text(_d[0], _d[1], _msg, fontsize=8, zorder=1))
    adjust_text(_ts, force_text=(0.5, 0.5), arrowprops=dict(arrowstyle='-|>', color='black', connectionstyle='arc3,rad=-.5', zorder=10), ax=_ax)
    lines = [Line2D([0], [0], color=_c, marker='o', mec='w', linestyle='', markersize=15) for _c in [viz.default_colors_dict['blue'], viz.default_colors_dict['blue_dark']]]
    plt.legend(lines, ['Min', 'Max'], labelcolor='#101628', bbox_to_anchor=(0.8, 1.0), loc='lower center', ncols=2, frameon=False, fontsize=14)
    plt.yticks(_xs[:nmax], ylabels[:nmax], fontsize=12)
    plt.xticks(fontsize=14)
    plt.xlabel('Score', fontsize=14)
    plt.gca().grid(axis='x')
    _msg = f'{title}'
    _fig.text(0, 1.0, _msg, fontweight='normal', fontsize=24, ha='left', color=viz.default_colors_dict['red'])
    subtitle = f"Scores are calculate from games' results in terms of score difference.\nMarker colors are clusters of teams with similar scores."
    _fig.text(0.0, 0.0, f'{subtitle}', size=11, color='#000000', ha='left')
    plt.tight_layout()
    _filename = tl.get_filename(f'soccer_{algo}_{fig_label}', lecture_id=lecture_id)
    _filename = None
    tl.savefig(plt, outfile=_filename, outdir=outdir_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 How is this related to the actual points attained by each team?

    Let's merge datasets of learned scores and official league standings
    """)
    return


@app.cell
def _(df_points, df_res):
    df_tot = df_res.merge(df_points,on=['node_label','competition_id']).sort_values(by='points_prg',ascending=False).reset_index(drop=True)
    df_tot.head()
    return (df_tot,)


@app.cell
def _(
    colors_1,
    competitionId2Name,
    df_tot,
    label_dict,
    lecture_id,
    np,
    outdir_fig,
    pearsonr,
    plt,
    spearmanr,
    tl,
):
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
    algo_1 = 'sr'
    _x = f'score_{algo_1}'
    _y = 'points_prg'
    _plot_linear_regression = True
    for i_1, (n, _g) in enumerate(df_tot.groupby(by='competition_id')):
        _spearman_coef = spearmanr(_g[_x], _g[_y])[0]
        _pearson_coef = pearsonr(_g[_x], _g[_y])[0]
        _msg = f'{competitionId2Name[n]}, sp = {_spearman_coef:.2f} | pr = {_pearson_coef:.2f}'
        _ax.scatter(_g[_x], _g[_y], c=colors_1[i_1], label=_msg)
        if _plot_linear_regression == True:
            _m, _b = np.polyfit(list(_g[_x]), list(_g[_y]), 1)
            _xmin, _xmax, _ymin, _ymax = plt.axis()
            _xs = np.linspace(_xmin, _xmax, 100)
            _ax.plot(_xs, _m * _xs + _b, ls='--', c=colors_1[i_1], alpha=0.8, lw=1)
    _ax.set_xlabel(f'Score {algo_1.upper()}')
    _ax.set_ylabel(label_dict[_y])
    plt.legend(loc='best', fontsize=10)
    _filename = tl.get_filename(f'soccer_{algo_1}_vs_points', lecture_id=lecture_id)
    _filename = None
    tl.savefig(plt, outfile=_filename, outdir=outdir_fig)
    return algo_1, i_1


@app.cell
def _(df_points_comp):
    df_points_comp[49]
    return


@app.cell
def _(
    adjust_text,
    colors_1,
    competitionId2Name,
    df_tot,
    i_1,
    np,
    pearsonr,
    plt,
    spearmanr,
):
    k_3 = 49
    _x = 'score_sr'
    _y = 'points_prg'
    _plot_linear_regression = True
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
    _g = df_tot[df_tot.competition_id == k_3]
    _spearman_coef = spearmanr(_g[_x], _g[_y])[0]
    _pearson_coef = pearsonr(_g[_x], _g[_y])[0]
    _msg = f'{competitionId2Name[k_3]}, sp = {_spearman_coef:.2f} | pr = {_pearson_coef:.2f}'
    _ax.scatter(_g[_x], _g[_y], c=colors_1[i_1], label=_msg)
    _ts = []
    for _idx, row in _g.iterrows():
        _msg = f'{row['node_label']}'
        _ts.append(_ax.text(row[_x], row[_y], _msg, fontsize=8, zorder=1))
    adjust_text(_ts, force_text=(0.5, 0.5), arrowprops=dict(arrowstyle='-|>', color='black', connectionstyle='arc3,rad=-.5', zorder=10), ax=_ax)
    if _plot_linear_regression == True:
        _m, _b = np.polyfit(list(_g[_x]), list(_g[_y]), 1)
        _xmin, _xmax, _ymin, _ymax = plt.axis()
        _xs = np.linspace(_xmin, _xmax, 100)
        _ax.plot(_xs, _m * _xs + _b, ls='--', c='grey', alpha=0.8, lw=1)
    _ax.set_xlabel(_x)
    _ax.set_ylabel(_y)
    plt.legend(loc='best', fontsize=10)
    return


@app.cell
def _(cols, df):
    k_4 = 49
    ref_team_name = 'North Carolina Courage'
    _cond1 = df[k_4].home_team == ref_team_name
    _cond2 = df[k_4].away_team == ref_team_name
    cond3 = df[k_4].home_score != df[k_4].away_score
    _mask = (_cond1 | _cond2) & cond3
    df[k_4][_mask][cols]
    return


@app.cell
def _(
    A,
    algo_1,
    colormap,
    lecture_id,
    model,
    nodeId2Label,
    np,
    outdir_fig,
    plt,
    tl,
    viz,
):
    k_5 = 49
    delta_x = 0.2
    _q = 0.75
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
    viz.plot_score_network(A[k_5], model[k_5].ranks, cm=colormap, ax=_ax, plot_labels=True, nodeId2Label=nodeId2Label[k_5])
    delta_ref = model[k_5].ranks.max() - model[k_5].ranks.min()
    delta_beta = np.log(_q / (1 - _q)) / (2 * model[k_5].beta)
    ys = np.linspace(model[k_5].ranks.min(), model[k_5].ranks.max(), 100)
    _xs = delta_x * np.ones(ys.shape[0])
    _ax.plot(_xs, ys, lw=1, color=viz.default_colors_dict['blue_sb_dark'])
    B = int(np.ceil(delta_ref / delta_beta))
    ys = np.arange(model[k_5].ranks.min(), model[k_5].ranks.min() + B * delta_beta, delta_beta)
    _xs = delta_x * np.ones(ys.shape[0])
    _ax.scatter(_xs, ys, lw=1, marker='_', color=viz.default_colors_dict['blue_sb_dark'])
    plt.tight_layout()
    _filename = tl.get_filename(f'soccer_{algo_1}_{k_5}_scores', lecture_id=lecture_id)
    _filename = None
    tl.savefig(plt, outfile=_filename, outdir=outdir_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Simulate games
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are ready to generate games from the main model parameters.
    """)
    return


@app.cell
def _(np):
    def get_H(s: np.ndarray, l: float=1):
        N = s.shape[0]
        _H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    _H[i, j] = 0.5 * (s[i] - s[j] - l)
        return _H

    return (get_H,)


@app.cell
def _(A, get_H, model, np, prng):
    k_6 = 49
    _beta = model[k_6].beta
    _H = get_H(model[k_6].ranks)
    _lambda_pois = np.exp(_beta * _H)
    np.fill_diagonal(_lambda_pois, 0)
    _c = np.sum(_lambda_pois) / np.sum(A[k_6])
    SAMPLE = 1000
    A_sim = np.array([prng.poisson(_lambda_pois) for s in np.arange(SAMPLE)])
    A_sim_avg = np.mean(A_sim, axis=0)
    np.fill_diagonal(A_sim_avg, 0)
    A_sim.shape
    return A_sim, A_sim_avg, SAMPLE, k_6


@app.cell
def _(A, A_sim_avg, k_6, model, np, plt, viz):
    _fig, _ax = plt.subplots(1, 2, figsize=(8, 4))
    _node_order = np.argsort(-model[k_6].ranks)
    viz.plot_matrix(A[k_6], ax=_ax[0], node_order=_node_order, title=f'GT data')
    viz.plot_matrix(A_sim_avg, ax=_ax[1], node_order=_node_order, title=f'Estimated average')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can select one example sample and check the data
    """)
    return


@app.cell
def _(A_sim, SAMPLE, colormap, model, nodeId2Label, np, plt, prng, viz):
    k_7 = 49
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
    _idx = prng.choice(np.arange(SAMPLE))
    viz.plot_score_network(A_sim[0], model[k_7].ranks, cm=colormap, ax=_ax, plot_labels=True, nodeId2Label=nodeId2Label[k_7], x_jit=0.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What happens if you change the hyperparameters?

    Note that this makes sense if you do not have a fixed schedule, and you want to generate that as well.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2.1 La Liga
    The top 3 teams are very close to each other. What is the probability that one of them wins the league if we were to simulate it n times?
    """)
    return


@app.cell
def _(A, get_H, model, np, prng):
    k_8 = 11
    _beta = model[k_8].beta
    _H = get_H(model[k_8].ranks)
    _lambda_pois = np.exp(_beta * _H)
    np.fill_diagonal(_lambda_pois, 0)
    _c = np.sum(_lambda_pois) / np.sum(A[k_8])
    SAMPLE_1 = 1000
    A_sim_1 = np.array([prng.poisson(_lambda_pois) for s in np.arange(SAMPLE_1)])
    A_sim_avg_1 = np.mean(A_sim_1, axis=0)
    np.fill_diagonal(A_sim_avg_1, 0)
    (A_sim_1.shape, _c, _beta)
    return A_sim_1, A_sim_avg_1, SAMPLE_1, k_8


@app.cell
def _(A, A_sim_1, A_sim_avg_1, SAMPLE_1, k_8, model, np, plt, prng, viz):
    _fig, _ax = plt.subplots(1, 3, figsize=(8, 4))
    _node_order = np.argsort(-model[k_8].ranks)
    viz.plot_matrix(A[k_8], ax=_ax[0], node_order=_node_order, title=f'GT data')
    _idx = prng.choice(np.arange(SAMPLE_1))
    viz.plot_matrix(A_sim_1[_idx], ax=_ax[1], node_order=_node_order, title=f'Example sample {_idx}')
    viz.plot_matrix(A_sim_avg_1, ax=_ax[2], node_order=_node_order, title=f'Estimated average')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, we can take every match in the schedule and simulate who wins.
    """)
    return


@app.cell
def _(k_8, np, pd, st):
    def get_simulated_games_df(df: pd.DataFrame, ranks: pd.DataFrame, nodeLabel2Id: dict, beta: float=1, competition_id: int=None):
        cols = ['home_team', 'away_team']
        score_diff = []
        df_new = df[cols].copy(deep=True)
        for _c in ('home_points', 'away_points'):
            df_new.loc[:, _c] = 0
        for _idx, rows in df_new.iterrows():
            i = nodeLabel2Id[rows['home_team']]
            j = nodeLabel2Id[rows['away_team']]
            s_i = ranks[i]
            s_j = ranks[j]
            p_ij = 1 / (1 + np.exp(-beta * (s_i - s_j)))
            r = st.bernoulli.rvs(p_ij, size=1)
            if r == 1:
                df_new.loc[_idx, 'home_points'] = 3
            elif r == 0:
                df_new.loc[_idx, 'away_points'] = 3
            else:
                print(f'r={r}')
        if competition_id is not None:
            df_new.loc[:, 'competiton_id'] = k_8
        return df_new

    return (get_simulated_games_df,)


@app.cell
def _(df, get_points, get_simulated_games_df, k_8, model, nodeLabel2Id):
    SAMPLE_2 = 100
    df_sim = [get_simulated_games_df(df[k_8], model[k_8].ranks, nodeLabel2Id[k_8], beta=model[k_8].beta, competition_id=k_8) for s in range(SAMPLE_2)]
    df_points_sim = [get_points(_d, competition_id=k_8) for _d in df_sim]
    return SAMPLE_2, df_points_sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check one particular (arbitrary) sample
    """)
    return


@app.cell
def _(SAMPLE_2, df_points_sim, np, prng):
    _idx = prng.choice(np.arange(SAMPLE_2))
    df_points_sim[_idx]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now count how many times in each of the simulated standing, one of the top 3 teams wins.
    """)
    return


@app.cell
def _(k_8, model, nodeId2Label, np):
    top3_ids = np.argsort(-model[k_8].ranks)[:3]
    top3_labels = [nodeId2Label[k_8][i] for i in top3_ids]
    (top3_labels, model[k_8].ranks[top3_ids])
    return (top3_labels,)


@app.cell
def _(df_points_sim, np, top3_labels):
    sim_ranks_top3 = np.zeros((len(top3_labels), len(top3_labels))).astype(int)
    for rid, ref_team in enumerate(top3_labels):
        for _df_tmp in df_points_sim:
            idx_sim = _df_tmp[_df_tmp.node_label == ref_team].index[0]
            if idx_sim < len(top3_labels):
                sim_ranks_top3[rid, idx_sim] = sim_ranks_top3[rid, idx_sim] + 1
    return (sim_ranks_top3,)


@app.cell
def _(pd, sim_ranks_top3, top3_labels):
    pd.DataFrame(sim_ranks_top3, columns = ['n_1st','n_2nd','n_3rd'], index=top3_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What are we missing?
    """)
    return


@app.cell
def _(df_points_comp, k_8):
    df_points_comp[k_8].iloc[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Depth of competition

    We can compare the statistics of the soccer league with results of other types of datasets.
    We take Table S2 [https://arxiv.org/pdf/1709.09002](of the SpringRank paper) for other datasets.
    """)
    return


@app.cell
def _(df_stats, np, pd, sr):
    _q = 0.75
    dataset = ['parakeet G1', 'parakeet G2', 'Asian elephants', 'Business', 'Computer Science', 'History', 'Village 1', 'Village 2']
    betas_S2 = np.array([2.7, 2.78, 2.33, 2.04, 2.23, 2.39, 1.98, 1.89])
    depth_S2 = np.array([2.604, 1.879, 3.0, 2.125, 2.423, 2.234, 3.618, 3.749])
    delta_level_S2 = np.array([np.log(_q / (1 - _q)) / (2 * _beta) for _beta in betas_S2])
    df_S2 = pd.DataFrame({'competition_id': [i + 100 for i in range(len(dataset))], 'competition_name': dataset, 'beta': betas_S2, 'depth': depth_S2, 'n_levels': sr.calculate_n_levels(depth_S2, betas_S2), 'delta_level': delta_level_S2})
    df_stats2 = pd.concat([df_stats, df_S2], axis=0).drop_duplicates()
    df_stats2
    return (df_stats2,)


@app.cell
def _(
    Line2D,
    algo_1,
    colors_1,
    df_stats2,
    lecture_id,
    np,
    outdir_fig,
    plt,
    tl,
):
    dataset_type = ['Soccer', 'Parakeet', 'Elephant', 'Faculty hiring', 'Villages']
    from matplotlib.patches import Patch
    _x = 'beta'
    _y = 'n_levels'
    color_plot = [colors_1[0] for i in range(5)] + [colors_1[1] for i in range(2)] + [colors_1[2] for i in range(1)] + [colors_1[3] for i in range(3)] + [colors_1[4] for i in range(2)]
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 4))
    _ax.bar(np.arange(len(df_stats2)), height=df_stats2[_y], color=color_plot, width=0.8, alpha=0.8)
    _ax.set_xlabel('Dataset')
    _ax.set_ylabel('Number of levels')
    x_tick_labels = df_stats2['competition_name'].values
    x_tick_labels[0] = 'FA WSL'
    _ax.set_xticks(np.arange(len(df_stats2)), labels=x_tick_labels, fontsize=8, rotation=60)
    legend_elements = [Line2D([0], [0], marker='o', color=colors_1[i], label=dataset_type[i], markerfacecolor=colors_1[i], markersize=10, lw=0) for i in np.arange(len(dataset_type))]
    _ax.legend(handles=legend_elements, loc='best')
    _ax.grid(axis='y')
    _filename = tl.get_filename(f'depth_competition_{algo_1}', lecture_id=lecture_id)
    _filename = None
    tl.savefig(plt, outfile=_filename, outdir=outdir_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Model selection

    How do we determine what scoring system is the best?

    **Homework**!
    """)
    return


if __name__ == "__main__":
    app.run()
