"""Research functions called by the published marimo research notebook.

This module contains no top-level analysis or command-line runner. The notebook
owns inputs, run controls, progress, caches, plots, and publication exports.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import brentq
from scipy.special import expit
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

CATEGORIES = {'ME':('Epee','Mens'), 'WE':('Epee','Womens'), 'MF':('Foil','Mens'),
              'WF':('Foil','Womens'), 'MS':('Sabre','Mens'), 'WS':('Sabre','Womens')}
REGIONS = ['Africa','Americas','Asia','Europe','Oceania']


def normalized_name(value):
    import unicodedata
    text = ''.join(c for c in unicodedata.normalize('NFKD',str(value)) if not unicodedata.combining(c))
    return ' '.join(sorted(''.join(c.lower() if c.isalnum() else ' ' for c in text).split()))


def official_regions(html):
    """Extract only public federation codes and confederations from FIE HTML."""
    from bs4 import BeautifulSoup
    payload = json.loads(BeautifulSoup(html,'html.parser').find('script',id='__NUXT_DATA__').string)
    labels = {'CAE':'Africa','CPE':'Americas','FCA':'Asia','CEE':'Europe','OFC':'Oceania'}
    result = {}
    for item in payload:
        if not isinstance(item,dict) or not {'zoneId','countries','code'}.issubset(item):
            continue
        confederation = payload[item['code']]
        if confederation not in labels:
            continue
        for reference in payload[item['countries']]:
            country = payload[reference]
            code = payload[country['nationality']]
            if code in result and result[code]['region'] != labels[confederation]:
                raise ValueError(f'Ambiguous FIE confederation for {code}')
            result[code] = {'region':labels[confederation], 'confederation':confederation,
                            'country':payload[country['country']]}
    if set(r['region'] for r in result.values()) != set(REGIONS):
        raise ValueError('FIE source did not contain all five confederations')
    return result


def prepare_bouts(frame):
    result = frame.dropna(subset=['Fencer A','Fencer B','Winner','Weapon','Gender','Age Category']).copy()
    result = result[result['Bout Status'].eq('Fenced') & result['Fencer A'].ne(result['Fencer B'])]
    result = result[result['Winner'].eq(result['Fencer A']) | result['Winner'].eq(result['Fencer B'])]
    result['Date'] = pd.to_datetime(result['Event Date'], format='mixed',errors='coerce')
    result['Block'] = result['Source URL'].fillna('').astype(str).str.split('#').str[0].str.rstrip('/')
    empty = result['Block'].eq('')
    result.loc[empty,'Block'] = result.loc[empty,['Event Name','City','Host Country','Event Date']].fillna('').astype(str).agg('|'.join,axis=1)
    result['Is Zonal'] = result['Event Name'].str.contains(
        r'zonal|zone|continental|(?:asian|european|african|pan.?american).*championship',case=False,na=False)
    return result


def beta_from_pairs(delta, wins, totals, legacy=False):
    """Nonnegative conditional win-likelihood estimate with explicit boundaries."""
    informative = np.abs(delta)>1e-12
    if not informative.any():
        return np.nan,'unidentified'
    weight = ((wins>0).astype(float)+((totals-wins)>0).astype(float)) if legacy else np.ones(len(delta))
    def equation(beta):
        return float(np.dot(weight*delta, wins-totals*expit(2*beta*delta)))
    if equation(0)<=1e-12:
        return 0.,'boundary_zero'
    upsets = np.where(delta>0,totals-wins,wins)
    if not np.any(upsets[informative]>0):
        return np.inf,'separated_infinite'
    high=1.
    while equation(high)>0 and high<1e12:
        high*=2
    return float(brentq(equation,0,high)),'finite'


def discrepancy(scores, fie_ranks):
    scores = np.asarray(scores,float); fie_ranks=np.asarray(fie_ranks,float)
    n=len(scores)
    if n<2:
        return np.full(n,np.nan)
    return (stats.rankdata(-scores,method='average')-stats.rankdata(fie_ranks,method='average'))/(n-1)


def spearman(scores,ranks):
    if len(scores)<3 or np.ptp(scores)==0 or np.ptp(ranks)==0:
        return np.nan
    return float(np.corrcoef(stats.rankdata(scores),stats.rankdata(ranks))[0,1])


@dataclass
class League:
    code: str
    names: np.ndarray
    countries: np.ndarray
    region: np.ndarray
    fie_rank: np.ndarray
    a: np.ndarray
    b: np.ndarray
    winner_a: np.ndarray
    event: np.ndarray
    date: np.ndarray
    stage: np.ndarray
    zonal: np.ndarray
    event_type: np.ndarray
    country_a: np.ndarray
    country_b: np.ndarray


def encode_leagues(bouts,rankings_root,regions,population='Senior events',exclude_zonal=False):
    selected=bouts[bouts['Age Category'].eq('Senior')] if population=='Senior events' else bouts
    if exclude_zonal:
        selected=selected[~selected['Is Zonal']]
    blocks=np.sort(selected['Block'].unique())
    block_ids={v:i for i,v in enumerate(blocks)}
    leagues={}
    for code,(weapon,gender) in CATEGORIES.items():
        g=selected[selected['Weapon'].eq(weapon)&selected['Gender'].eq(gender)]
        names=np.array(sorted(set(g['Fencer A'])|set(g['Fencer B'])))
        ids={name:i for i,name in enumerate(names)}
        affiliations=pd.concat([g[['Fencer A','Country A']].rename(columns={'Fencer A':'name','Country A':'country'}),
                                g[['Fencer B','Country B']].rename(columns={'Fencer B':'name','Country B':'country'})]).dropna()
        affiliation=affiliations.groupby('name')['country'].agg(lambda v:v.value_counts().index[0]).to_dict()
        countries=np.array([affiliation.get(name,'Unknown') for name in names],dtype=object)
        ranks=np.full(len(names),np.nan)
        path=Path(rankings_root)/f'{code}-detailed-ranking-2026.csv'
        if path.exists():
            official=pd.read_csv(path); official['key']=official['Name'].map(normalized_name)
            official=official[~official['key'].duplicated(keep=False)]
            lookup=pd.to_numeric(official.set_index('key')['Rank'],errors='coerce').to_dict()
            keys=pd.Series(names).map(normalized_name)
            ambiguous=keys.duplicated(keep=False).to_numpy()
            ranks=np.array([lookup.get(key,np.nan) for key in keys],float);ranks[ambiguous]=np.nan
        leagues[code]=League(code,names,countries,np.array([regions.get(c,{}).get('region','Unknown') for c in countries]),
            ranks,g['Fencer A'].map(ids).to_numpy(int),g['Fencer B'].map(ids).to_numpy(int),g['Winner'].eq(g['Fencer A']).to_numpy(),
            g['Block'].map(block_ids).to_numpy(int),g['Date'].to_numpy(),g['Stage'].fillna('Unknown').to_numpy(),
            g['Is Zonal'].to_numpy(),g['Event Name'].fillna('').to_numpy(),g['Country A'].fillna('Unknown').to_numpy(),g['Country B'].fillna('Unknown').to_numpy())
    return leagues,blocks


def fit_league(league,minimum=5,event_weights=None,indices=None):
    n=len(league.names)
    weight=np.ones(len(league.a)) if event_weights is None else np.asarray(event_weights)[league.event]
    if indices is not None:
        weight=weight*np.asarray(indices,bool)
    counts=np.bincount(league.a,weights=weight,minlength=n)+np.bincount(league.b,weights=weight,minlength=n)
    keep=(counts[league.a]>=minimum)&(counts[league.b]>=minimum)&(weight>0)
    u=np.where(league.winner_a,league.a,league.b)
    v=np.where(league.winner_a,league.b,league.a)
    A=sparse.coo_matrix((weight[keep],(u[keep],v[keep])),shape=(n,n)).tocsr()
    nonempty=np.asarray((A+A.T).sum(axis=1)).ravel()>0
    ids=np.flatnonzero(nonempty); A=A[ids][:,ids]
    if len(ids)<5:
        raise ValueError('fewer_than_five_eligible_athletes')
    components,labels=connected_components(A+A.T,directed=False)
    largest=labels==np.argmax(np.bincount(labels));ids=ids[largest];A=A[largest][:,largest]
    out=np.asarray(A.sum(axis=1)).ravel();incoming=np.asarray(A.sum(axis=0)).ravel()
    laplacian=sparse.diags(out+incoming,dtype=float)-A-A.T;rhs=out-incoming
    scores=np.zeros(len(ids))
    with warnings.catch_warnings():
        warnings.simplefilter('error',sparse.linalg.MatrixRankWarning)
        scores[1:]=spsolve(laplacian[1:,1:],rhs[1:])
    scores-=scores.mean()
    if not np.isfinite(scores).all() or np.max(np.abs(laplacian@scores-rhs))>1e-7:
        raise ValueError('rank_solver_failed')
    pairs=sparse.triu(A+A.T,k=1).tocoo()
    delta=scores[pairs.row]-scores[pairs.col]
    wins=np.asarray(A[pairs.row,pairs.col]).ravel()
    beta,status=beta_from_pairs(delta,wins,pairs.data)
    legacy,legacy_status=beta_from_pairs(delta,wins,pairs.data,True)
    matched=np.isfinite(league.fie_rank[ids]); rho=spearman(scores[matched],league.fie_rank[ids][matched])
    depth=float(np.ptp(scores))
    upset=np.where(delta>0,pairs.data-wins,wins)
    informative=np.abs(delta)>1e-12
    frame=pd.DataFrame({'category':league.code,'athlete_index':ids,'name':league.names[ids],'country':league.countries[ids],
                        'region':league.region[ids],'score':scores,'retained_bouts':(out+incoming).astype(int),
                        'fie_rank':league.fie_rank[ids]})
    frame['model_rank']=stats.rankdata(-scores,method='min').astype(int)
    frame['D']=np.nan
    frame.loc[matched,'D']=discrepancy(scores[matched],league.fie_rank[ids][matched])
    summary={'category':league.code,'beta':beta,'beta_status':status,'legacy_beta':legacy,'legacy_beta_status':legacy_status,
             'depth':depth,'levels':2*beta*depth/np.log(3),'iqr_levels':2*beta*float(np.subtract(*np.percentile(scores,[75,25])))/np.log(3),
             'rho':rho,'matched':int(matched.sum()),'athletes':len(ids),'bouts':int(A.sum()),
             'components_before_filter':int(components),'observed_upset_rate':float(upset[informative].sum()/pairs.data[informative].sum()),
             'tied_score_bout_fraction':float(pairs.data[~informative].sum()/pairs.data.sum())}
    return frame,summary


def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);adjusted=np.empty(len(p));last=0.
    for i,index in enumerate(order):
        last=max(last,(len(p)-i)*p[index]);adjusted[index]=min(last,1.)
    return adjusted


def bootstrap_tables(point,draws,minimum_valid=.9):
    """Centered bootstrap contrasts; replicates are not independent observations."""
    codes=list(CATEGORIES); draws=np.asarray(draws,float)
    valid=np.isfinite(draws).all(axis=1);sample=draws[valid]
    result={'requested':len(draws),'joint_valid':int(valid.sum()),'valid_fraction':float(valid.mean()),
            'formal_inference_enabled':bool(valid.mean()>=minimum_valid),'intervals':[],'pairs':[]}
    if len(sample)<20:
        return result
    center=np.asarray(point,float); errors=sample-center
    low,high=np.quantile(sample,[.025,.975],axis=0)
    result['intervals']=[{'category':c,'estimate':float(center[i]),'low':float(low[i]),'high':float(high[i])} for i,c in enumerate(codes)]
    C=np.zeros((5,6));C[:,0]=-1;C[np.arange(5),np.arange(1,6)]=1
    covariance=np.cov(sample,rowvar=False);inv=np.linalg.pinv(C@covariance@C.T)
    contrast=C@center; statistic=float(contrast@inv@contrast)
    null_contrasts=errors@C.T
    null_statistics=np.einsum('ij,jk,ik->i',null_contrasts,inv,null_contrasts)
    result['omnibus']={'statistic':statistic,'p':float((1+(null_statistics>=statistic).sum())/(len(sample)+1))} if result['formal_inference_enabled'] else None
    pvalues=[]
    for i in range(6):
        for j in range(i+1,6):
            d=sample[:,i]-sample[:,j];estimate=center[i]-center[j]
            p=float((1+(np.abs(d-estimate)>=abs(estimate)).sum())/(len(d)+1))
            lower,upper=np.quantile(d,[.025,.975]);pvalues.append(p)
            result['pairs'].append({'a':codes[i],'b':codes[j],'difference':float(estimate),'low':float(lower),'high':float(upper),
                                    'p':p if result['formal_inference_enabled'] else None})
    adjusted=holm(pvalues)
    for row,value in zip(result['pairs'],adjusted):
        row['p_holm']=float(value) if result['formal_inference_enabled'] else None
    return result


def regional_contrast(frame):
    means=frame.groupby('region')['D'].mean()
    needed=['Africa','Americas','Asia','Europe']
    if any(r not in means or not np.isfinite(means[r]) for r in needed):
        return np.nan
    return float((means['Africa']+means['Americas']-means['Asia']-means['Europe'])/2)


def region_means(frame):
    return frame.groupby('region')['D'].mean().reindex(REGIONS).to_numpy(float)


_BOOTSTRAP_STATE=None


def _init_bootstrap(leagues,n_events):
    global _BOOTSTRAP_STATE
    _BOOTSTRAP_STATE=(leagues,n_events)


def _bootstrap_one(seed):
    leagues,n_events=_BOOTSTRAP_STATE
    rng=np.random.default_rng(seed);weights=np.bincount(rng.integers(0,n_events,n_events),minlength=n_events)
    betas=[];rhos=[];sizes=[];region_frames=[];failures=[];regional=[]
    for code in CATEGORIES:
        try:
            frame,summary=fit_league(leagues[code],5,weights)
            betas.append(summary['beta'] if summary['beta_status'] in ['finite','boundary_zero'] else np.nan)
            rhos.append(summary['rho']);sizes.append(summary['athletes'])
            region_frames.append(frame.dropna(subset=['D']))
            regional.append(region_means(frame.dropna(subset=['D'])).tolist())
            if not np.isfinite(betas[-1]):failures.append(code+':'+summary['beta_status'])
        except Exception as exc:
            betas.append(np.nan);rhos.append(np.nan);sizes.append(0);regional.append([np.nan]*5);failures.append(code+':'+str(exc))
    contrast=regional_contrast(pd.concat(region_frames,ignore_index=True)) if region_frames else np.nan
    return betas,rhos,sizes,contrast,failures,regional


def run_bootstrap(leagues,blocks,replicates=2000,seed=42,workers=4,progress=None,checkpoint=None):
    """Invoked from a notebook cell; callback exposes every batch to the UI."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import sys
    seeds=[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(replicates)]
    digest=hashlib.sha256()
    for code,league in leagues.items():
        digest.update(code.encode())
        for value in [league.a,league.b,league.winner_a,league.event,league.fie_rank]:digest.update(value.tobytes())
        digest.update('|'.join(league.region).encode())
    signature=digest.hexdigest()
    results=[]
    if checkpoint and Path(checkpoint).exists():
        cached=json.loads(Path(checkpoint).read_text(encoding='utf-8'))
        if cached['seed']==seed and cached['replicates']==replicates and cached.get('signature')==signature:
            results=cached['results']
    def save():
        if checkpoint:
            path=Path(checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'seed':seed,'replicates':replicates,'signature':signature,'results':results},allow_nan=True),encoding='utf-8');temp.replace(path)
    if sys.platform=='emscripten' or workers==1:
        _init_bootstrap(leagues,len(blocks));iterator=map(_bootstrap_one,seeds[len(results):]);pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init_bootstrap,initargs=(leagues,len(blocks)))
        iterator=pool.map(_bootstrap_one,seeds[len(results):],chunksize=5)
    try:
        for result in iterator:
            results.append(result)
            if progress:progress(len(results),replicates)
            if len(results)%25==0:save()
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        save()
    return {'beta':np.array([r[0] for r in results]),'rho':np.array([r[1] for r in results]),
            'sizes':np.array([r[2] for r in results]),'regional_contrast':np.array([r[3] for r in results]),
            'failures':[r[4] for r in results],'regional_means':np.array([r[5] for r in results]),'signature':signature,'seed':seed}


def fantasy_probability(score_a,beta_a,score_b,beta_b,mean_a=0.,mean_b=0.):
    if not np.isfinite([score_a,beta_a,score_b,beta_b,mean_a,mean_b]).all():
        raise ValueError('A finite identified beta and scores are required')
    return float(expit(2*beta_a*(score_a-mean_a)-2*beta_b*(score_b-mean_b)))


def biography_features(path,leagues,cutoff):
    """Use published birth dates, never birthdays inferred from license numbers."""
    table=pd.read_csv(path,low_memory=False)
    table['key']=table['Fencer'].map(normalized_name)+'|'+table['Country'].fillna('Unknown')
    duplicate=table['key'].duplicated(keep=False)
    lookup=table[~duplicate].set_index('key')
    cutoff=pd.Timestamp(cutoff)
    results={}
    for code,league in leagues.items():
        rows=[]
        for name,country in zip(league.names,league.countries):
            key=normalized_name(name)+'|'+str(country)
            if key in lookup.index:
                record=lookup.loc[key]
                date=record.get('FIE Date of Birth')
                if pd.isna(date):date=record.get('Directory Date of Birth')
                dob=pd.to_datetime(date,errors='coerce')
                age=(cutoff-dob).days/365.2425 if pd.notna(dob) else np.nan
                hand=record.get('FIE Handedness','Unknown')
                hand=hand if hand in ['Left','Right','Ambidextrous'] else 'Unknown'
                height=pd.to_numeric(record.get('FIE Height cm'),errors='coerce')
            else:dob=pd.NaT;age=np.nan;hand='Unknown';height=np.nan
            rows.append({'name':name,'country':country,'dob':dob,'age':age if 8<=age<=90 else np.nan,
                         'hand':hand,'height':height if pd.notna(height) and 100<=height<=230 else np.nan})
        results[code]=pd.DataFrame(rows)
    return results,{'athletes':len(table),'statuses':table['Biography Import Status'].value_counts().to_dict(),
                    'height_present':int(table['FIE Height cm'].notna().sum()),'club_present':int(table['FIE Club'].notna().sum())}


def discrepancy_features(leagues,fits,biographies):
    all_rows=[]
    for code,league in leagues.items():
        frame=fits[code][0].dropna(subset=['D']).copy()
        frame=frame.merge(biographies[code],on=['name','country'],how='left',validate='one_to_one')
        events=[];bouts=[];pool_share=[];zonal_share=[];satellite_share=[]
        for index in frame['athlete_index']:
            mask=(league.a==index)|(league.b==index)
            events.append(len(np.unique(league.event[mask])));bouts.append(int(mask.sum()))
            pool_share.append(float(np.mean(league.stage[mask]=='Pool')))
            zonal_share.append(float(np.mean(league.zonal[mask])))
            satellite_share.append(float(np.mean(['satellite' in name.lower() for name in league.event_type[mask]])))
        frame['events']=events;frame['recorded_bouts']=bouts;frame['pool_share']=pool_share
        frame['zonal_share']=zonal_share;frame['satellite_share']=satellite_share
        all_rows.append(frame)
    return pd.concat(all_rows,ignore_index=True)


def clustered_adjustment(frame,participation=False):
    """OLS with country-clustered sandwich covariance; associations, not causal effects."""
    data=frame[frame['region'].isin(REGIONS)].copy()
    data['region']=pd.Categorical(data['region'],categories=['Europe','Africa','Americas','Asia','Oceania'])
    data['hand']=pd.Categorical(data['hand'].fillna('Unknown'),categories=['Right','Left','Ambidextrous','Unknown'])
    columns=['region','category','hand']
    X=pd.get_dummies(data[columns],drop_first=True,dtype=float)
    X['age_missing']=data['age'].isna().astype(float)
    X['age']=(data['age'].fillna(data['age'].median())-25)/10
    if participation:
        X['log_events']=np.log1p(data['events']);X['log_bouts']=np.log1p(data['recorded_bouts'])
        for column in ['pool_share','zonal_share','satellite_share']:X[column]=data[column]
    X=X.loc[:,X.nunique()>1];X.insert(0,'Intercept',1.)
    matrix=X.to_numpy(float);y=data['D'].to_numpy(float)
    inverse=np.linalg.pinv(matrix.T@matrix);coefficients=inverse@matrix.T@y;residual=y-matrix@coefficients
    meat=np.zeros_like(inverse);groups=data['country'].to_numpy();unique=np.unique(groups)
    for group in unique:
        score=matrix[groups==group].T@residual[groups==group];meat+=np.outer(score,score)
    rank=np.linalg.matrix_rank(matrix);g=len(unique);n=len(y)
    covariance=inverse@meat@inverse*(g/(g-1))*((n-1)/(n-rank))
    se=np.sqrt(np.maximum(np.diag(covariance),0));critical=stats.t.ppf(.975,g-1)
    rows=[]
    for name,estimate,error in zip(X.columns,coefficients,se):
        rows.append({'term':name,'estimate':float(estimate),'se':float(error),'low':float(estimate-critical*error),
                     'high':float(estimate+critical*error),'p':float(2*stats.t.sf(abs(estimate/error),g-1)) if error>0 else None})
    c=np.zeros(len(X.columns))
    for name,value in [('region_Africa',.5),('region_Americas',.5),('region_Asia',-.5)]:
        if name in X.columns:c[X.columns.get_loc(name)]=value
    estimate=float(c@coefficients);error=float(np.sqrt(max(c@covariance@c,0)))
    return {'n':n,'countries':g,'participation_adjusted':participation,'coefficients':rows,
            'contrast':{'estimate':estimate,'low':estimate-critical*error,'high':estimate+critical*error,
                        'p':float(2*stats.t.sf(abs(estimate/error),g-1)) if error>0 else None},
            'caveat':'Country-clustered conditional associations; not a causal estimate or a full propagation of ranking uncertainty.'}


def discrepancy_prediction(frame,seed=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.model_selection import GroupKFold,GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    data=frame.copy()
    identities=data['name'].map(normalized_name)
    country_counts=data.assign(identity=identities).groupby('identity')['country'].nunique()
    ambiguous=set(country_counts[country_counts>1].index)
    excluded=int(identities.isin(ambiguous).sum())
    data=data.loc[~identities.isin(ambiguous)].reset_index(drop=True)
    groups=data['country'].fillna('Unknown').to_numpy();y=data['D'].to_numpy()
    categorical=['category','region','hand'];numeric=['age','events','recorded_bouts','pool_share','zonal_share','satellite_share']
    cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
    predictions={name:np.full(len(data),np.nan) for name in ['zero','ridge','gradient_boosting']};folds=[]
    for fold,(train,test) in enumerate(cv.split(data,y,groups)):
        assert not set(groups[train])&set(groups[test])
        assert not set(data.iloc[train]['name'].map(normalized_name))&set(data.iloc[test]['name'].map(normalized_name))
        predictions['zero'][test]=0
        for name,estimator,grid in [
            ('ridge',Ridge(solver='lsqr'),{'model__alpha':[1.,10.,100.]}),
            ('gradient_boosting',GradientBoostingRegressor(random_state=seed,n_estimators=120,max_depth=2),{'model__min_samples_leaf':[15,40]})]:
            transform=ColumnTransformer([
                ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric),
                ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
            pipeline=Pipeline([('features',transform),('model',estimator)])
            inner=GroupKFold(n_splits=min(3,len(np.unique(groups[train]))))
            search=GridSearchCV(pipeline,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1)
            search.fit(data.iloc[train],y[train],groups=groups[train]);predictions[name][test]=search.predict(data.iloc[test])
        folds.append({'fold':fold+1,'train_n':len(train),'test_n':len(test),'train_countries':len(np.unique(groups[train])),
                      'test_countries':sorted(set(groups[test])),'country_overlap':0,'athlete_overlap':0})
    results=[]
    for model,prediction in predictions.items():
        results.append({'model':model,'n':len(y),'mae':float(mean_absolute_error(y,prediction)),'r2':float(r2_score(y,prediction))})
    regional=[]
    for region in sorted(data['region'].unique()):
        mask=data['region'].eq(region).to_numpy()
        for model,prediction in predictions.items():
            regional.append({'region':region,'model':model,'n':int(mask.sum()),'mae':float(mean_absolute_error(y[mask],prediction[mask]))})
    return {'metrics':results,'regional_errors':regional,'folds':folds,'excluded_cross_country_identity_rows':excluded,
            'identity_policy':'Conservatively exclude normalized names recorded under multiple countries from grouped prediction.',
            'target':'Contemporaneous normalized rank discrepancy',
            'validation':'Country-grouped outer folds and country-grouped inner tuning; no future-rank forecasting claim.'}


def probability_metrics(y,p):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return {'n':len(y),'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            'brier':float(np.mean((y-p)**2)),
            'accuracy':float(np.mean(np.where(np.isclose(p,.5,atol=1e-12),.5,(p>.5)==y)))}


def calibration(y,p):
    y=np.asarray(y);p=np.asarray(p);rows=[]
    bins=np.minimum((p*10).astype(int),9)
    for index in range(10):
        mask=bins==index
        if mask.any():rows.append({'bin':index,'n':int(mask.sum()),'mean_probability':float(p[mask].mean()),'observed_win_fraction':float(y[mask].mean())})
    return rows


def rolling_bout_prediction(leagues,biographies,regions,seed=42,progress=None):
    """Three event-time holdouts; fit ranks, beta, imputation and models anew per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    bounds={}
    for league in leagues.values():
        valid=~np.isnat(league.date)
        for event in np.unique(league.event[valid]):
            dates=league.date[valid&(league.event==event)]
            lo,hi=dates.min(),dates.max()
            if int(event) in bounds:lo=min(lo,bounds[int(event)][0]);hi=max(hi,bounds[int(event)][1])
            bounds[int(event)]=(lo,hi)
    ordered=sorted(bounds,key=lambda e:(bounds[e][1],e))
    folds=[];predictions=[];excluded=[];total_test=0
    for fold,(start,end) in enumerate([(.4,.6),(.6,.8),(.8,1.)],1):
        left=int(len(ordered)*start);right=int(len(ordered)*end)
        test_events=ordered[left:right]
        test_start=min(bounds[e][0] for e in test_events)
        train_events=[e for e in ordered[:left] if bounds[e][1]<test_start]
        assert not set(train_events)&set(test_events)
        train_end=max(bounds[e][1] for e in train_events)
        assert train_end<test_start
        for code,league in leagues.items():
            valid=~np.isnat(league.date)
            train=valid&np.isin(league.event,train_events)
            test=valid&np.isin(league.event,test_events);total_test+=int(test.sum())
            try:
                fitted,summary=fit_league(league,5,indices=train)
                if summary['beta_status'] not in ['finite','boundary_zero']:
                    raise ValueError('No finite training-only beta')
                score=np.full(len(league.names),np.nan);score[fitted['athlete_index']]=fitted['score']
                available=np.isfinite(score[league.a])&np.isfinite(score[league.b])
                train &= available;test &= available
                if train.sum()<50 or test.sum()<10:
                    raise ValueError('Too few covered training or test bouts')
                bio=biographies[code]
                dob=pd.to_datetime(bio['dob']).to_numpy(dtype='datetime64[ns]')
                hand=bio['hand'].fillna('Unknown').to_numpy()
                left_hand=np.array([1. if h=='Left' else 0. if h=='Right' else .5 for h in hand])
                missing_hand=(hand=='Unknown').astype(float)
                train_age=np.concatenate([(league.date[train]-dob[league.a[train]])/np.timedelta64(1,'D')/365.2425,
                                          (league.date[train]-dob[league.b[train]])/np.timedelta64(1,'D')/365.2425])
                sensible=train_age[(train_age>=8)&(train_age<=90)]
                median=float(np.nanmedian(sensible)) if len(sensible) else 25.
                def design(mask):
                    a=league.a[mask];b=league.b[mask];date=league.date[mask]
                    age_a=(date-dob[a])/np.timedelta64(1,'D')/365.2425
                    age_b=(date-dob[b])/np.timedelta64(1,'D')/365.2425
                    age_a=np.where((age_a>=8)&(age_a<=90),age_a,np.nan)
                    age_b=np.where((age_b>=8)&(age_b<=90),age_b,np.nan)
                    eta=2*summary['beta']*(score[a]-score[b])
                    columns=[eta,(np.nan_to_num(age_a,nan=median)-np.nan_to_num(age_b,nan=median))/10,
                             np.isnan(age_a).astype(float)-np.isnan(age_b).astype(float),
                             left_hand[a]-left_hand[b],missing_hand[a]-missing_hand[b]]
                    ra=np.array([regions.get(c,{}).get('region','Unknown') for c in league.country_a[mask]])
                    rb=np.array([regions.get(c,{}).get('region','Unknown') for c in league.country_b[mask]])
                    for region in REGIONS+['Unknown']:columns.append((ra==region).astype(float)-(rb==region).astype(float))
                    return np.column_stack(columns)
                X=design(train);y=league.winner_a[train].astype(int)
                model=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=1.,fit_intercept=False,max_iter=1000,random_state=seed))
                model.fit(np.vstack([X,-X]),np.concatenate([y,1-y]))
                Z=design(test);probability=model.predict_proba(Z)[:,1]
                swapped=model.predict_proba(-Z)[:,1]
                assert np.max(np.abs(probability+swapped-1))<1e-10
                baseline=expit(Z[:,0]);outcomes=league.winner_a[test].astype(int)
                for event,outcome,p0,p1 in zip(league.event[test],outcomes,baseline,probability):
                    predictions.append((fold,code,int(event),int(outcome),float(p0),float(p1)))
                folds.append({'fold':fold,'category':code,'training_bouts':int(train.sum()),'covered_test_bouts':int(test.sum()),
                              'all_test_bouts':int((valid&np.isin(league.event,test_events)).sum()),
                              'train_end':str(pd.Timestamp(train_end).date()),'test_start':str(pd.Timestamp(test_start).date()),
                              'event_overlap':0,'training_beta':summary['beta'],'swap_error':float(np.max(np.abs(probability+swapped-1)))})
            except Exception as exc:
                excluded.append({'fold':fold,'category':code,'reason':str(exc)})
            if progress:progress((fold-1)*6+list(CATEGORIES).index(code)+1,18)
    if not predictions:
        return {'status':'unavailable','reason':'No covered chronological test bouts','excluded':excluded}
    frame=pd.DataFrame(predictions,columns=['fold','category','event','outcome','springrank','biography'])
    metrics=[]
    for code in ['All']+list(CATEGORIES):
        data=frame if code=='All' else frame[frame['category'].eq(code)]
        if data.empty:continue
        for model in ['springrank','biography']:
            metrics.append({'category':code,'model':model,**probability_metrics(data['outcome'],data[model])})
    y=frame['outcome'].to_numpy();base=frame['springrank'].to_numpy();bio=frame['biography'].to_numpy()
    difference=(y-base)**2-(y-bio)**2
    event_errors=pd.DataFrame({'event':frame['event'],'difference':difference}).groupby('event')['difference'].agg(['sum','count'])
    rng=np.random.default_rng(seed+1);draws=[]
    for _ in range(2000):
        ids=rng.integers(0,len(event_errors),len(event_errors));sample=event_errors.iloc[ids]
        draws.append(float(sample['sum'].sum()/sample['count'].sum()))
    low,high=np.quantile(draws,[.025,.975])
    return {'status':'complete','metrics':metrics,'folds':folds,'excluded':excluded,
            'covered_test_bouts':len(frame),'all_test_bouts':total_test,'coverage':len(frame)/total_test,
            'calibration':{'springrank':calibration(y,base),'biography':calibration(y,bio)},
            'brier_improvement':{'estimate':float(difference.mean()),'low':float(low),'high':float(high),
                                 'interpretation':'Positive means lower held-out Brier error with biographies.'},
            'traits_caveat':'Birth dates are time-invariant; handedness comes from current profile records and is assumed stable. Clubs/coaches are excluded.',
            'test_prediction_hash':hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()}
