"""Paired zonal-point counterfactuals invoked by visible marimo cells."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import research


def attach_points(frame, table):
    """Keep exactly the original matched cohort; use average ties in both point orders."""
    eligible=frame.dropna(subset=['D']).copy()
    lookup=table.loc[~table['key'].duplicated(keep=False)].set_index('key')
    eligible['key']=eligible['name'].map(research.normalized_name)
    columns=['official_total','zonal_points','points_without_zonal','rank_from_total_average_ties','rank_without_zonal_average_ties']
    merged=eligible.merge(lookup[columns],left_on='key',right_index=True,how='left',validate='one_to_one')
    assert len(merged)==len(eligible) and merged[columns].notna().all().all()
    n=len(merged)
    assert n>1
    sr=merged['score'].rank(ascending=False,method='average')
    merged['D_original']=merged['D']
    merged['D_total_ties']=(sr-merged['official_total'].rank(ascending=False,method='average'))/(n-1)
    merged['D_without_zonal']=(sr-merged['points_without_zonal'].rank(ascending=False,method='average'))/(n-1)
    merged['D_change']=merged['D_without_zonal']-merged['D_total_ties']
    merged['global_rank_change']=merged['rank_without_zonal_average_ties']-merged['rank_from_total_average_ties']
    merged['zonal_fraction']=np.divide(merged['zonal_points'],merged['official_total'],out=np.zeros(n),where=merged['official_total'].to_numpy()>0)
    return merged


def summarize(frames):
    joined=pd.concat(frames,ignore_index=True)
    contrasts={}
    for column in ['D_original','D_total_ties','D_without_zonal','D_change']:
        data=joined.copy();data['D']=data[column]
        contrasts[column]=research.regional_contrast(data)
    baseline=contrasts['D_total_ties']
    contrasts['fraction_reduction']=1-contrasts['D_without_zonal']/baseline if abs(baseline)>1e-12 else np.nan
    correlations=[]
    for frame in frames:
        before=research.spearman(frame['score'],frame['official_total'].rank(ascending=False))
        after=research.spearman(frame['score'],frame['points_without_zonal'].rank(ascending=False))
        correlations.append({'category':frame['category'].iloc[0],'n':len(frame),'rho_total_ties':before,
                             'rho_without_zonal':after,'rho_change':after-before})
    return contrasts,correlations


_STATE=None


def _init(leagues,tables,n_events):
    global _STATE
    _STATE=leagues,tables,n_events


def _draw(seed):
    leagues,tables,n_events=_STATE
    rng=np.random.default_rng(seed)
    weights=np.bincount(rng.integers(0,n_events,n_events),minlength=n_events)
    frames=[]
    try:
        for code,league in leagues.items():
            fitted,_=research.fit_league(league,5,weights)
            frames.append(attach_points(fitted,tables[code]))
        contrast,correlation=summarize(frames)
        regional=[]
        for region in research.REGIONS:
            subset=pd.concat(frames,ignore_index=True)
            subset=subset[subset['region'].eq(region)]
            regional.append([float(subset[c].mean()) if len(subset) else None
                             for c in ['D_total_ties','D_without_zonal','D_change']])
        return {'contrasts':contrast,'correlations':correlation,'sizes':[len(f) for f in frames],
                'regional':regional,'failure':None}
    except Exception as exc:
        return {'failure':type(exc).__name__+': '+str(exc)}


def paired_bootstrap(leagues,tables,blocks,replicates=2000,seed=42,workers=4,checkpoint=None,progress=None):
    """Refit shared tournament draws and evaluate both point orders in each draw."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    digest=hashlib.sha256(Path(__file__).read_bytes())
    digest.update(Path(research.__file__).read_bytes())
    for code,league in leagues.items():
        digest.update(code.encode())
        for values in [league.a,league.b,league.winner_a,league.event,league.fie_rank]:digest.update(values.tobytes())
        digest.update('|'.join(league.region).encode())
        digest.update(tables[code][['key','official_total','zonal_points','points_without_zonal']].to_csv(index=False).encode())
    signature=digest.hexdigest();results=[]
    if checkpoint and Path(checkpoint).exists():
        cached=json.loads(Path(checkpoint).read_text(encoding='utf-8'))
        if cached.get('signature')==signature and cached.get('seed')==seed and cached.get('replicates')==replicates:
            results=cached['results']
    seeds=[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(replicates)]
    def save():
        if checkpoint:
            path=Path(checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'signature':signature,'seed':seed,'replicates':replicates,'results':results},allow_nan=False),encoding='utf-8');tmp.replace(path)
    if workers==1:
        _init(leagues,tables,len(blocks));pool=None;iterator=map(_draw,seeds[len(results):])
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init,initargs=(leagues,tables,len(blocks)))
        iterator=pool.map(_draw,seeds[len(results):],chunksize=5)
    try:
        for result in iterator:
            results.append(result)
            if progress:progress(len(results),replicates)
            if len(results)%25==0:save()
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        save()
    return {'results':results,'signature':signature,'replicates':replicates,'seed':seed}


def interval(estimate,draws,requested):
    draws=np.asarray(draws,float);draws=draws[np.isfinite(draws)]
    enabled=len(draws)>=.9*requested
    if len(draws)<20:
        return {'estimate':float(estimate),'valid':len(draws),'formal_inference_enabled':False,'low':None,'high':None,'p':None}
    lo,hi=np.quantile(draws,[.025,.975])
    return {'estimate':float(estimate),'low':float(lo),'high':float(hi),'valid':len(draws),
            'formal_inference_enabled':enabled,
            'p':float((1+(np.abs(draws-estimate)>=abs(estimate)).sum())/(1+len(draws))) if enabled else None}
