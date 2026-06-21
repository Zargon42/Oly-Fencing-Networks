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

    return np, pd


@app.cell
def _():
    import sys
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
        df_points.loc[:, 'points_prg'] = (df_points['points'] / df_points['n_matches']).map(lambda x: np.round(_x, 2))
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
    df_plot_dist = df_res.groupby(by='competition_id')[metric].agg(['describe']).droplevel(0, axis=1).reset_index().sort_values(by='competition_id', key=lambda x: _x.map(compId2sort))
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
            p_ij = 1 / (1 + np.exp(-_beta * (s_i - s_j)))
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
