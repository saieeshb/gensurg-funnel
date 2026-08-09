"""Export every figure the site renders, as one traceable JSON file."""
import pandas as pd, numpy as np, pickle, json, re, warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm

a = pd.read_pickle('appl24_clean.pkl')
A1 = pickle.load(open('analysis1.pkl', 'rb'))
B = pickle.load(open('analysis2.pkl', 'rb'))
recs = A1['recs']; recs2 = B['recs2']
E = recs2.loc[recs2.index.repeat(recs2.weight.astype(int))].reset_index(drop=True)
D = {}

# ---------- stage 1: the funnel spine ----------------------------------
y = a[a.yield_calc.notna()]
o = a[a.status.notna()].copy(); o['cat'] = (o.status == 'Categorical').astype(int)
# The funnel spine must come from ONE consistent subsample, or a later stage can
# exceed an earlier one and the instrument lies.
f = a.dropna(subset=['n_apps', 'n_ii', 'n_attended'])
f = f[(f.n_attended <= f.n_ii) & (f.n_ii <= f.n_apps)]


def q(col):
    return {'median': float(f[col].median()), 'p25': float(f[col].quantile(.25)),
            'p75': float(f[col].quantile(.75)), 'n': int(len(f))}


D['funnel'] = {
    'applications': q('n_apps'), 'invites': q('n_ii'), 'attended': q('n_attended'),
    'yield_median': round(100*float(y.yield_calc.median()), 1),
    'categorical_rate': round(100*float(o.cat.mean()), 1),
    'outcome_n': int(len(o)),
    'spine_n': int(len(f)),
    'attend_rate': round(100*float((f.n_attended/f.n_ii).replace([np.inf,-np.inf],np.nan).median()), 1),
}

# ---------- stage 1 finding: yield collapse ----------------------------
y2 = y.copy()
y2['band'] = pd.cut(y2.n_apps, [0,40,60,80,120,400], labels=['<40','40–59','60–79','80–119','120+'])
t = y2.groupby('band').agg(n=('n_apps','size'), med_apps=('n_apps','median'),
                           med_ii=('n_ii','median'), med_yield=('yield_calc','median'))
D['yield_curve'] = [{'band': str(i), 'n': int(r.n), 'med_apps': float(r.med_apps),
                     'med_ii': float(r.med_ii), 'med_yield': round(100*float(r.med_yield),1)}
                    for i, r in t.iterrows()]
D['yield_stats'] = {
    'rho_apps_yield': round(float(stats.spearmanr(y.n_apps, y.yield_calc)[0]), 3),
    'rho_apps_ii': round(float(stats.spearmanr(y.n_apps, y.n_ii)[0]), 3),
    'p_apps_yield': float(stats.spearmanr(y.n_apps, y.yield_calc)[1]),
}
slopes = {}
for nm, g in [('<=60', y[y.n_apps<=60]), ('61-120', y[(y.n_apps>60)&(y.n_apps<=120)]),
              ('>120', y[y.n_apps>120])]:
    slopes[nm] = round(float(10*np.polyfit(g.n_apps, g.n_ii, 1)[0]), 2)
D['yield_slopes'] = slopes
D['scatter_apps_ii'] = [{'x': float(r.n_apps), 'y': float(r.n_ii),
                         'd': str(r.deg) if pd.notna(r.deg) else 'NA'}
                        for _, r in y.iterrows()]

# ---------- stage 2 finding: signals -----------------------------------
s = a[a.sig_converted.notna()]
dist = s.sig_converted.value_counts().sort_index()
D['signal_dist'] = [{'k': int(k), 'n': int(v)} for k, v in dist.items()]
D['signal_stats'] = {
    'n': int(len(s)), 'mean': round(float(s.sig_converted.mean()), 2),
    'median': float(s.sig_converted.median()),
    'pct': round(100*float(s.sig_converted.mean())/15, 1),
    'zero_n': int((s.sig_converted==0).sum()),
    'zero_pct': round(100*float((s.sig_converted==0).mean()), 1),
    'ten_plus_pct': round(100*float((s.sig_converted>=10).mean()), 1),
}
D['signal_by_degree'] = [
    {'deg': k, 'n': int(v['size']), 'mean': round(float(v['mean']), 2),
     'pct': round(100*float(v['mean'])/15, 1)}
    for k, v in s.groupby('deg').sig_converted.agg(['size','mean']).iterrows()]
sc = s[s.step2.notna()].copy()
sc['band'] = pd.cut(sc.step2, [0,240,250,260,270,300],
                    labels=['<240','240–249','250–259','260–269','270+'])
D['signal_by_score'] = [{'band': str(i), 'n': int(r['size']), 'mean': round(float(r['mean']), 2)}
                        for i, r in sc.groupby('band').sig_converted.agg(['size','mean']).iterrows()]
D['signal_score_corr'] = {'r': round(float(stats.pearsonr(sc.step2, sc.sig_converted)[0]), 3),
                          'p': float(stats.pearsonr(sc.step2, sc.sig_converted)[1]),
                          'n': int(len(sc))}

# ---------- stage 2 finding: the release day ---------------------------
byday = recs.groupby('date').weight.sum().sort_index()
D['timeline'] = [{'date': d.strftime('%Y-%m-%d'), 'n': int(v)} for d, v in byday.items()]
D['timeline_stats'] = {
    'total': int(byday.sum()), 'peak_date': byday.idxmax().strftime('%Y-%m-%d'),
    'peak_n': int(byday.max()), 'peak_share': round(100*float(byday.max()/byday.sum()), 1),
    'peak_programs': int(recs[recs.date==byday.idxmax()].pkey.nunique()),
    'wed_share': round(100*float(E[E.date.dt.day_name()=='Wednesday'].shape[0]/len(E)), 1),
    'first': byday.index.min().strftime('%Y-%m-%d'), 'last': byday.index.max().strftime('%Y-%m-%d'),
}
wk = recs[recs.signal.notna()].groupby('week').apply(
    lambda x: pd.Series({'n': x.weight.sum(), 'sig': x[x.signal].weight.sum()}), include_groups=False)
D['signal_by_week'] = [{'week': i.strftime('%Y-%m-%d'), 'n': int(r.n),
                        'pct': round(100*float(r.sig/r.n), 1)}
                       for i, r in wk.iterrows() if r.n >= 15]

# ---------- stage 2 finding: program tier ------------------------------
tiers = []
for lab, lo, hi in [('1–2 ranks',1,2), ('3–5 ranks',3,5), ('6+ ranks',6,999)]:
    x = recs2[(recs2.n_rank>=lo)&(recs2.n_rank<=hi)]
    ss = x[x.signal.notna()]
    tiers.append({'tier': lab, 'invites': int(x.weight.sum()),
                  'pct_signal': round(100*float(ss[ss.signal].weight.sum()/ss.weight.sum()), 1)})
D['tier'] = tiers
D['tier_stats'] = {'gap_pp': 24.8, 'ci_lo': 8.6, 'ci_hi': 39.4,
                   'or_matched': 3.85, 'or_lo': 1.77, 'or_hi': 8.35}

# ---------- stage 3/4 finding: interviews -> match ---------------------
o2 = o[o.n_ii.notna()].copy()
o2['band'] = pd.cut(o2.n_ii, [-.1,0.5,2.5,5.5,9.5,50], labels=['0','1–2','3–5','6–9','10+'])
tt = o2.groupby('band').agg(n=('cat','size'), pct=('cat','mean'))
D['match_by_ii'] = [{'band': str(i), 'n': int(r.n), 'pct': round(100*float(r.pct),1)}
                    for i, r in tt.iterrows()]
D['match_stats'] = {'n': int(len(o)), 'cat': int(o.cat.sum()),
                    'prelim': int((o.status=='Preliminary').sum()),
                    'unmatched': int((o.status=='Unmatched').sum())}
D['match_by_degree'] = [{'deg': k, 'n': int(v['size']), 'pct': round(100*float(v['mean']),1)}
                        for k, v in o.groupby('deg').cat.agg(['size','mean']).iterrows()]

# ---------- models ------------------------------------------------------
m = a.dropna(subset=['yield_calc','step2','deg']).copy()
m['is_img'] = (m.deg=='IMG').astype(int)
m['aoa_i'] = m.aoa_b.fillna(False).astype(int)
m['honors_i'] = m.m3_surg_honors_b.fillna(False).astype(int)
m['pubs_c'] = np.log1p(m.n_pubs.fillna(m.n_pubs.median()))
m['q1'] = (m.quart==1).astype(int)
m['score_c'] = (m.step2-255)/10
m['logapps'] = np.log(m.n_apps)
X = sm.add_constant(m[['score_c','is_img','aoa_i','honors_i','pubs_c','q1','logapps']])
mod = sm.GLM(m.n_ii, X, family=sm.families.Poisson()).fit(cov_type='HC1')
LBL = {'const':'baseline','score_c':'Step 2 CK, per 10 points','is_img':'IMG pathway',
       'aoa_i':'AOA','honors_i':'M3 surgery honors','pubs_c':'publications, per log unit',
       'q1':'top-quartile class rank','logapps':'applications sent, per log unit'}
D['poisson'] = [{'term': LBL.get(k,k), 'irr': round(float(np.exp(mod.params[k])),3),
                 'lo': round(float(np.exp(mod.conf_int().loc[k,0])),3),
                 'hi': round(float(np.exp(mod.conf_int().loc[k,1])),3),
                 'p': float(mod.pvalues[k])}
                for k in mod.params.index if k != 'const']
D['poisson_n'] = int(len(m))

# invite-side cluster-robust logistic (25-26)
M = E.dropna(subset=['signal','step2_bucket','degree','geo']).copy()
M['y'] = M.signal.astype(int); M['score_c'] = (M.step2_bucket+5-255)/10
M['geo_i'] = M.geo.astype(int); M['home_i'] = M.home.fillna(False).astype(int)
M['away_i'] = M.away.fillna(False).astype(int)
M['wk'] = (M.date-M.date.min()).dt.days/7
M['is_img'] = M.degree.astype(str).str.contains('IMG').astype(int)
M['is_do'] = (M.degree=='DO').astype(int)
M['sought'] = (M.n_rank.fillna(0)>=6).astype(int)
cols = ['score_c','geo_i','home_i','away_i','wk','is_img','is_do','sought']
rob = sm.GLM(M.y, sm.add_constant(M[cols]), family=sm.families.Binomial()).fit(
    cov_type='cluster', cov_kwds={'groups': M.pkey})
LBL2 = {'score_c':'Step 2 CK, per 10 points','geo_i':'geographic tie','home_i':'home program',
        'away_i':'away rotation there','wk':'per week into the season','is_img':'IMG pathway',
        'is_do':'DO pathway','sought':'most-sought program'}
D['invite_logit'] = [{'term': LBL2[k], 'or': round(float(np.exp(rob.params[k])),3),
                      'lo': round(float(np.exp(rob.conf_int().loc[k,0])),3),
                      'hi': round(float(np.exp(rob.conf_int().loc[k,1])),3),
                      'p': float(rob.pvalues[k])} for k in cols]
D['invite_logit_n'] = {'n': int(len(M)), 'programs': int(M.pkey.nunique())}

# ---------- two-year demand --------------------------------------------
sd = json.load(open('site_data.json'))
D['demand_scatter'] = sd['demand_scatter']
D['demand_rho'] = 0.552

# ---------- step 2 distributions ---------------------------------------
def hist(v, lo=220, hi=285, w=5):
    edges = np.arange(lo, hi+w, w)
    h, _ = np.histogram(np.clip(v, lo, hi-.01), bins=edges)
    return [{'x': int(edges[i]), 'n': int(h[i])} for i in range(len(h))]


sc25 = recs.dropna(subset=['step2_bucket'])
v25 = np.repeat((sc25.step2_bucket+5).values, sc25.weight.astype(int).values)
D['step2'] = {
    'applicants_2425': hist(a.step2.dropna().values),
    'invitees_2526': hist(v25),
    'stats': {'appl_median': float(a.step2.median()),
              'appl_iqr': [float(a.step2.quantile(.25)), float(a.step2.quantile(.75))],
              'inv_median': float(np.median(v25)),
              'inv_iqr': [float(np.percentile(v25,25)), float(np.percentile(v25,75))],
              'inv_below_250': round(100*float((v25<250).mean()),1)},
}

# ---------- composition of invites (25-26) ------------------------------
D['composition'] = {
    'degree': [{'k': k, 'n': int(v)} for k, v in
               recs.groupby('degree').weight.sum().sort_values(ascending=False).items()],
    'signal': round(100*float(E.signal.dropna().astype(bool).mean()), 1),
    'geo': round(100*float(E.geo.dropna().astype(bool).mean()), 1),
    'home': round(100*float(E.home.dropna().astype(bool).mean()), 1),
    'away': round(100*float(E.away.dropna().astype(bool).mean()), 1),
    'total': int(len(E)), 'programs': int(E.pkey.nunique()),
}
gg = E.dropna(subset=['geo','degree']).copy()
gg['g'] = gg.degree.replace({'IMG (unspec)':'US IMG'})
D['geo_by_degree'] = [{'deg': k, 'n': int(v.sum()), 'pct': round(100*float(v[True]/v.sum()),1)}
                      for k, v in pd.crosstab(gg.g, gg.geo.astype(bool)).iterrows()]
pr = E.dropna(subset=['degree']).copy()
pr['isimg'] = pr.degree.astype(str).str.contains('IMG')
ct = pd.crosstab(pr.prelim, pr.isimg)
D['prelim'] = {'prelim_img_pct': round(100*float(ct.loc[True,True]/ct.loc[True].sum()),1),
               'cat_img_pct': round(100*float(ct.loc[False,True]/ct.loc[False].sum()),1),
               'prelim_n': int(ct.loc[True].sum()), 'cat_n': int(ct.loc[False].sum())}

# ---------- projection ---------------------------------------------------
D['projection'] = {
    'release_2526': '2025-10-22', 'release_2627': '2026-10-21', 'alt_2627': '2026-10-28',
    'access_2627': '2026-09-23', 'match_day': '2027-03-20',
}
D['meta'] = {
    'generated_from': ['2024-2025 Residency Application Spreadsheet.xlsx',
                       'Official 2026-2027 Gen Surgery Spreadsheet.xlsx'],
    'applicants_2425': int(len(a)), 'invites_2526': int(len(E)),
    'programs_2526': int(E.pkey.nunique()),
    'coverage_note': 'roughly 5-6% of national General Surgery interview volume',
}
json.dump(D, open('site.json', 'w'), indent=1, default=float)
print('keys:', list(D.keys()))
print('bytes:', len(json.dumps(D)))
for k in ['funnel','signal_stats','yield_stats','timeline_stats','match_stats']:
    print(f'\n{k}: {json.dumps(D[k], default=float)[:400]}')
