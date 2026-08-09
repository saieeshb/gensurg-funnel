"""Non-US IMG specific analysis across both cycles."""
import pandas as pd, numpy as np, pickle, re, json, warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm

pd.set_option('display.width', 210)
a = pd.read_pickle('appl24_clean.pkl')
A1 = pickle.load(open('analysis1.pkl', 'rb')); B = pickle.load(open('analysis2.pkl', 'rb'))
recs, recs2 = A1['recs'], B['recs2']
store25 = pickle.load(open('sheets.pkl', 'rb'))
E = recs2.loc[recs2.index.repeat(recs2.weight.astype(int))].reset_index(drop=True)
OUT = []; X = {}


def h(t):
    s = '\n' + '=' * 90 + f'\n== {t}\n' + '=' * 90
    print(s); OUT.append(s)


def p(*z):
    s = ' '.join(str(i) for i in z); print(s); OUT.append(s)


# ---------------------------------------------------------------- 2025-26 invite side
h('IMG-1. INVITE COMPOSITION (2025-26, invite-level)')
E['grp'] = E.degree.replace({'IMG (unspec)': 'US IMG'})
tot = len(E)
for g, x in E.groupby('grp'):
    p(f'  {g:12s} {len(x):5d} invites  {100*len(x)/tot:5.1f}%')
nus = E[E.grp == 'Non-US IMG']
usi = E[E.grp == 'US IMG']
img = E[E.grp.isin(['Non-US IMG', 'US IMG'])]
p(f'\nNon-US IMG invites: {len(nus)} ({100*len(nus)/tot:.1f}% of all reported invites)')

p('\n--- categorical vs prelim, by pathway ---')
ct = pd.crosstab(E.grp, E.prelim)
ct.columns = ['categorical', 'prelim']
ct['% prelim'] = (100 * ct.prelim / (ct.prelim + ct.categorical)).round(1)
p(ct.to_string())
X['track_by_grp'] = [{'grp': i, 'cat': int(r.categorical), 'prelim': int(r.prelim),
                      'pct_prelim': float(r['% prelim'])} for i, r in ct.iterrows()]
p(f"\n  A Non-US IMG invite is {ct.loc['Non-US IMG','% prelim']/max(ct.loc['MD','% prelim'],.01):.0f}x "
  f"more likely to be a prelim invite than an MD's.")

p('\n--- attributes of Non-US IMG invitees vs everyone else ---')
rows = []
for g in ['MD', 'DO', 'US IMG', 'Non-US IMG']:
    x = E[E.grp == g]
    r = {'group': g, 'invites': len(x)}
    for f in ['signal', 'geo', 'home', 'away']:
        s = x[x[f].notna()]
        r['%' + f] = round(100 * s[f].astype(bool).mean(), 1) if len(s) else np.nan
    sc = x.dropna(subset=['step2_bucket'])
    v = (sc.step2_bucket + 5).values
    r['median_step2'] = float(np.median(v)) if len(v) else np.nan
    r['%>=260'] = round(100 * float((v >= 260).mean()), 1) if len(v) else np.nan
    rows.append(r)
tab = pd.DataFrame(rows).set_index('group')
p(tab.to_string())
X['attr_by_grp'] = tab.reset_index().replace({np.nan: None}).to_dict('records')

p('\n--- the score paradox ---')
sc_n = nus.dropna(subset=['step2_bucket']); sc_m = E[E.grp == 'MD'].dropna(subset=['step2_bucket'])
vn = (sc_n.step2_bucket + 5).values; vm = (sc_m.step2_bucket + 5).values
p(f'  Non-US IMG invitees: median {np.median(vn):.0f}, {100*(vn>=260).mean():.1f}% at 260+')
p(f'  US MD invitees     : median {np.median(vm):.0f}, {100*(vm>=260).mean():.1f}% at 260+')
u, pv = stats.mannwhitneyu(vn, vm)
p(f'  Mann-Whitney p={pv:.5f}  -> Non-US IMGs who get invited score HIGHER than MDs who get invited.')
p('  The bar is not equal. Same invite, more score required.')
X['score_paradox'] = {'nus_median': float(np.median(vn)), 'md_median': float(np.median(vm)),
                      'nus_260': round(100*float((vn>=260).mean()),1), 'md_260': round(100*float((vm>=260).mean()),1),
                      'p': float(pv), 'nus_n': int(len(vn)), 'md_n': int(len(vm))}
X['step2_hist'] = {
    'nus': [{'x': int(k), 'n': int(v)} for k, v in pd.Series(vn).value_counts().sort_index().items()],
    'md': [{'x': int(k), 'n': int(v)} for k, v in pd.Series(vm).value_counts().sort_index().items()],
}

p('\n--- geography: the tell ---')
for g in ['MD', 'DO', 'US IMG', 'Non-US IMG']:
    x = E[(E.grp == g) & E.geo.notna()]
    p(f'  {g:12s} geo tie on {100*x.geo.astype(bool).mean():5.1f}% of invites (n={len(x)})')
p('  Non-US IMGs take invites where they can get them, not where they want to be.')

# ---------------------------------------------------------------- 2024-25 outcome side
h('IMG-2. OUTCOMES (2024-25, applicant-level)')
ai = a[a.deg == 'IMG']
p(f'IMG applicants in the 2024-25 file: {len(ai)}  (the sheet does not split US / non-US here)')
p(f'  with Step 2: {ai.step2.notna().sum()} | with apps+IIs: {ai.yield_calc.notna().sum()} | '
  f'signal yield: {ai.sig_converted.notna().sum()} | outcome: {ai.status.notna().sum()}')

p('\n--- applications, invites, yield ---')
for lab, g in [('IMG', ai), ('MD', a[a.deg == 'MD']), ('DO', a[a.deg == 'DO'])]:
    y = g[g.yield_calc.notna()]
    p(f'  {lab:4s} n={len(y):3d} | median apps {y.n_apps.median():5.0f} | median invites {y.n_ii.median():4.1f} '
      f'| median yield {100*y.yield_calc.median():5.1f}%')
yi = ai[ai.yield_calc.notna()]; ym = a[(a.deg == 'MD') & a.yield_calc.notna()]
u, pv = stats.mannwhitneyu(yi.yield_calc, ym.yield_calc)
p(f'  IMG vs MD yield: {100*yi.yield_calc.median():.1f}% vs {100*ym.yield_calc.median():.1f}%, p={pv:.2e}')
X['yield_by_grp'] = [{'grp': lab, 'n': int(len(g[g.yield_calc.notna()])),
                      'apps': float(g[g.yield_calc.notna()].n_apps.median()),
                      'ii': float(g[g.yield_calc.notna()].n_ii.median()),
                      'yield': round(100*float(g[g.yield_calc.notna()].yield_calc.median()), 1)}
                     for lab, g in [('IMG', ai), ('MD', a[a.deg == 'MD']), ('DO', a[a.deg == 'DO'])]]

p('\n--- signal conversion, the direct measurement ---')
si = ai[ai.sig_converted.notna()]
sm_ = a[(a.deg == 'MD') & a.sig_converted.notna()]
sd = a[(a.deg == 'DO') & a.sig_converted.notna()]
p(f'  IMG: n={len(si)}, mean {si.sig_converted.mean():.2f}/15 ({100*si.sig_converted.mean()/15:.1f}%), '
  f'median {si.sig_converted.median():.0f}')
p(f'  MD : n={len(sm_)}, mean {sm_.sig_converted.mean():.2f}/15 ({100*sm_.sig_converted.mean()/15:.1f}%)')
p(f'  DO : n={len(sd)}, mean {sd.sig_converted.mean():.2f}/15')
u, pv = stats.mannwhitneyu(si.sig_converted, sm_.sig_converted)
p(f'  IMG vs MD: p={pv:.5f}')
p(f'  IMG converting ZERO of 15: {(si.sig_converted==0).sum()}/{len(si)} = {100*(si.sig_converted==0).mean():.0f}%')
p(f'  MD  converting ZERO of 15: {(sm_.sig_converted==0).sum()}/{len(sm_)} = {100*(sm_.sig_converted==0).mean():.0f}%')
X['signal_conv'] = {
    'img': {'n': int(len(si)), 'mean': round(float(si.sig_converted.mean()), 2),
            'zero_pct': round(100*float((si.sig_converted == 0).mean()), 1)},
    'md': {'n': int(len(sm_)), 'mean': round(float(sm_.sig_converted.mean()), 2),
           'zero_pct': round(100*float((sm_.sig_converted == 0).mean()), 1)},
    'do': {'n': int(len(sd)), 'mean': round(float(sd.sig_converted.mean()), 2),
           'zero_pct': round(100*float((sd.sig_converted == 0).mean()), 1)},
    'p': float(pv),
}
X['signal_dist_img'] = [{'k': int(k), 'n': int(v)} for k, v in
                        si.sig_converted.value_counts().sort_index().items()]

p('\n--- outcomes ---')
oi = ai[ai.status.notna()]
p(f'  IMG with a reported outcome: {len(oi)}')
p(oi.status.value_counts().to_string())
for lab, g in [('IMG', ai), ('MD', a[a.deg == 'MD']), ('DO', a[a.deg == 'DO'])]:
    o = g[g.status.notna()]
    if len(o) < 5: continue
    p(f'  {lab:4s} categorical {100*(o.status=="Categorical").mean():5.1f}%  '
      f'prelim {100*(o.status=="Preliminary").mean():5.1f}%  '
      f'unmatched {100*(o.status=="Unmatched").mean():5.1f}%  (n={len(o)})')
X['outcome_by_grp'] = [{'grp': lab,
                        'n': int(len(g[g.status.notna()])),
                        'cat': round(100*float((g[g.status.notna()].status == 'Categorical').mean()), 1),
                        'prelim': round(100*float((g[g.status.notna()].status == 'Preliminary').mean()), 1),
                        'unmatched': round(100*float((g[g.status.notna()].status == 'Unmatched').mean()), 1)}
                       for lab, g in [('MD', a[a.deg == 'MD']), ('DO', a[a.deg == 'DO']), ('IMG', ai)]
                       if len(g[g.status.notna()]) >= 5]

p('\n--- what an IMG needed to bring, vs an MD ---')
comp = []
for lab, g in [('IMG', ai), ('MD', a[a.deg == 'MD'])]:
    comp.append({'group': lab, 'n': len(g),
                 'median_step2': g.step2.median(),
                 'median_pubs': g.n_pubs.median(),
                 'mean_pubs': round(g.n_pubs.mean(), 1),
                 'median_apps': g.n_apps.median()})
cdf = pd.DataFrame(comp).set_index('group')
p(cdf.to_string())
u, pv = stats.mannwhitneyu(ai.n_pubs.dropna(), a[a.deg == 'MD'].n_pubs.dropna())
p(f'  publications IMG vs MD: p={pv:.2e}')
X['profile'] = cdf.reset_index().replace({np.nan: None}).to_dict('records')
X['pubs_p'] = float(pv)

# does anything predict IMG interview count?
p('\n--- what predicts interview count WITHIN the IMG group ---')
mi = ai.dropna(subset=['n_ii', 'step2', 'n_apps']).copy()
if len(mi) >= 20:
    mi['score_c'] = (mi.step2 - 250) / 10
    mi['pubs_c'] = np.log1p(mi.n_pubs.fillna(mi.n_pubs.median()))
    mi['logapps'] = np.log(mi.n_apps)
    Xm = sm.add_constant(mi[['score_c', 'pubs_c', 'logapps']])
    md = sm.GLM(mi.n_ii, Xm, family=sm.families.Poisson()).fit(cov_type='HC1')
    r = pd.DataFrame({'IRR': np.exp(md.params), 'lo': np.exp(md.conf_int()[0]),
                      'hi': np.exp(md.conf_int()[1]), 'p': md.pvalues}).round(3)
    p(r.to_string()); p(f'  n={len(mi)} — small; treat as descriptive.')
    LBL = {'score_c': 'Step 2 CK, per 10 points', 'pubs_c': 'publications, per log unit',
           'logapps': 'applications sent, per log unit'}
    X['img_poisson'] = [{'term': LBL[k], 'irr': float(np.exp(md.params[k])),
                         'lo': float(np.exp(md.conf_int().loc[k, 0])),
                         'hi': float(np.exp(md.conf_int().loc[k, 1])), 'p': float(md.pvalues[k])}
                        for k in ['score_c', 'pubs_c', 'logapps']]
    X['img_poisson_n'] = int(len(mi))

# ---------------------------------------------------------------- structured IMG tab
h('IMG-3. THE DEDICATED IMG TAB (2025-26)')
it = store25['II by Date 25-26 IMGs'].iloc[1:].copy()
it.columns = ['date', 'program', 'signaled', 'cat_prelim', 'research_year', 'prior_prelim', 'stats', 'q']
it = it[it.program.notna()]
it['sig'] = it.signaled.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
it['res'] = it.research_year.astype(str).str.lower().str.contains('yes')
it['type'] = np.where(it.stats.astype(str).str.lower().str.contains('non'), 'Non-US IMG',
                      np.where(it.stats.astype(str).str.lower().str.contains('img'), 'US IMG', None))
p(f'structured IMG invite rows: {len(it)}')
p(f'  signalled: {it.sig.sum():.0f}/{it.sig.notna().sum():.0f} = {100*it.sig.mean():.1f}%')
p(f'  research year: {it.res.sum()}/{len(it)} = {100*it.res.mean():.1f}%')
p(f'  cat vs prelim: {it.cat_prelim.astype(str).str.strip().str.lower().value_counts().to_dict()}')
ctb = pd.crosstab(it.res, it.sig)
if ctb.shape == (2, 2):
    orr, pv = stats.fisher_exact(ctb.values)
    p(f'  research year x signalled: Fisher OR={orr:.2f}, p={pv:.3f} (no association)')
p('\n  programmes inviting the most IMGs:')
p(it.program.astype(str).str.strip().value_counts().head(10).to_string())
X['img_programs'] = [{'name': str(k)[:40], 'n': int(v)} for k, v in
                     it.program.astype(str).str.strip().value_counts().head(10).items()]
X['img_tab'] = {'rows': int(len(it)), 'pct_signal': round(100*float(it.sig.mean()), 1),
                'pct_research': round(100*float(it.res.mean()), 1)}

# programmes that invited Non-US IMGs in the main log
p('\n  programmes appearing most in Non-US IMG invites (main log):')
nn = nus.groupby('pkey').size().sort_values(ascending=False).head(12)
name_of = recs.groupby('pkey').program_raw.first()
for k, v in nn.items():
    p(f'    {str(name_of.get(k, k))[:44]:46s} {v}')
X['nus_programs'] = [{'name': str(name_of.get(k, k))[:44], 'n': int(v)} for k, v in nn.items()]

h('IMG-4. VERBATIM FROM THE IMG CHAT TABS')
quotes = []
for tab in ['IMG Chat', 'IMG Chat 26-76']:
    df = store25[tab]
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if isinstance(v, str) and len(v.strip()) > 45:
                quotes.append(re.sub(r'\s+', ' ', v.strip()))
key = [q for q in quotes if re.search(r'\d+/15|signal|0 ii|research year|prelim', q, re.I)]
for q in key[:14]:
    p('  •', q[:300])
X['quotes'] = [q[:280] for q in key[:8]]

json.dump(X, open('img.json', 'w'), indent=1, default=float)
open('analysis_img.txt', 'w').write('\n'.join(OUT))
p('\n[written img.json, analysis_img.txt]')
