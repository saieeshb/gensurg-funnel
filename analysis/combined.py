"""Combined two-cycle analysis: 24-25 applicant outcomes + 25-26 invite composition."""
import pandas as pd, numpy as np, pickle, re, warnings, json
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm

pd.set_option('display.width', 230)
a = pd.read_pickle('appl24.pkl')
A1 = pickle.load(open('analysis1.pkl', 'rb'))
B = pickle.load(open('analysis2.pkl', 'rb'))
recs = A1['recs']
store24 = pickle.load(open('sheets24.pkl', 'rb'))
store25 = pickle.load(open('sheets.pkl', 'rb'))
OUT = []
EXPORT = {}


def h(t):
    s = '\n' + '=' * 92 + f'\n== {t}\n' + '=' * 92
    print(s); OUT.append(s)


def p(*x):
    s = ' '.join(str(i) for i in x); print(s); OUT.append(s)


# ======================================================================
h('19. DATA CLEANING ON THE 24-25 APPLICANT FILE')
bad = a[(a.step2 < 180) & a.step2.notna()]
p(f'implausible Step 2 values (<180): {len(bad)} -> {sorted(bad.step2.dropna().tolist())}  [dropped]')
a.loc[a.step2 < 180, 'step2'] = np.nan
hi = a[a.n_apps > 300]
p(f'application counts >300: {len(hi)} -> {sorted(hi.n_apps.dropna().astype(int).tolist())}  [kept, plausible for IMGs]')
p(f'yield >1.0 (more invites than applications): {(a.yield_calc > 1).sum()}  [none expected]')
a.loc[a.yield_calc > 1, 'yield_calc'] = np.nan
p(f'\nusable applicant records: {len(a)}')
p(f'  with Step 2: {a.step2.notna().sum()} | with apps+IIs: {a.yield_calc.notna().sum()} | '
  f'with signal yield: {a.sig_converted.notna().sum()} | with outcome: {a.status.notna().sum()}')

# ======================================================================
h('20. THE DIRECT MEASUREMENT: WHAT 15 SIGNALS ACTUALLY BUY')
s = a[a.sig_converted.notna()]
p(f'n = {len(s)} applicants reporting signal yield out of 15')
p(f'  mean {s.sig_converted.mean():.2f}/15 converted ({100*s.sig_converted.mean()/15:.1f}%)')
p(f'  median {s.sig_converted.median():.0f}/15 | IQR {s.sig_converted.quantile(.25):.0f}-{s.sig_converted.quantile(.75):.0f}')
dist = s.sig_converted.value_counts().sort_index()
p('\n  distribution of signals converted:')
for k, v in dist.items():
    p(f'    {int(k):2d}/15  {v:3d}  {"#"*int(v/1.5)}')
p(f'\n  applicants converting ZERO signals: {(s.sig_converted==0).sum()} ({100*(s.sig_converted==0).mean():.1f}%)')
p(f'  converting >=10: {(s.sig_converted>=10).sum()} ({100*(s.sig_converted>=10).mean():.1f}%)')

p('\n--- 20a. Signal conversion by pathway ---')
t = s.groupby('deg').sig_converted.agg(['size','mean','median'])
t['pct_of_15'] = 100*t['mean']/15
p(t.round(2).to_string())
md, img = s[s.deg=='MD'].sig_converted, s[s.deg=='IMG'].sig_converted
if len(img) >= 5:
    u, pv = stats.mannwhitneyu(md, img)
    p(f'  MD vs IMG: {md.mean():.2f} vs {img.mean():.2f} signals converted, Mann-Whitney p={pv:.4f}')
do = s[s.deg=='DO'].sig_converted
u, pv = stats.mannwhitneyu(md, do)
p(f'  MD vs DO : {md.mean():.2f} vs {do.mean():.2f}, p={pv:.4f}')

p('\n--- 20b. Does the signal conversion rate depend on your score? ---')
sc = s[s.step2.notna()]
r, pv = stats.pearsonr(sc.step2, sc.sig_converted)
p(f'  corr(Step 2, signals converted) = {r:+.3f}, p={pv:.5f}, n={len(sc)}')
sc2 = sc.copy(); sc2['band'] = pd.cut(sc2.step2, [0,240,250,260,270,300],
                                      labels=['<240','240-249','250-259','260-269','270+'])
p(sc2.groupby('band').sig_converted.agg(['size','mean']).round(2).to_string())
EXPORT['signal_by_band'] = sc2.groupby('band').sig_converted.agg(['size','mean']).round(2).reset_index().to_dict('records')
EXPORT['signal_dist'] = [{'converted': int(k), 'n': int(v)} for k, v in dist.items()]

# ======================================================================
h('21. THE YIELD CURVE — does applying to more programs help?')
y = a[a.yield_calc.notna() & a.n_apps.notna() & a.n_ii.notna()]
p(f'n = {len(y)} applicants with both application and invite counts')
p(f'  applications: median {y.n_apps.median():.0f}, IQR {y.n_apps.quantile(.25):.0f}-{y.n_apps.quantile(.75):.0f}, max {y.n_apps.max():.0f}')
p(f'  invites     : median {y.n_ii.median():.0f}, IQR {y.n_ii.quantile(.25):.0f}-{y.n_ii.quantile(.75):.0f}, max {y.n_ii.max():.0f}')
p(f'  yield       : median {100*y.yield_calc.median():.1f}%, IQR {100*y.yield_calc.quantile(.25):.1f}-{100*y.yield_calc.quantile(.75):.1f}%')
y2 = y.copy()
y2['band'] = pd.cut(y2.n_apps, [0,40,60,80,120,400],
                    labels=['<40','40-59','60-79','80-119','120+'])
tab = y2.groupby('band').agg(n=('n_apps','size'), med_apps=('n_apps','median'),
                             med_ii=('n_ii','median'), mean_ii=('n_ii','mean'),
                             med_yield=('yield_calc','median'))
tab['med_yield'] = (100*tab.med_yield).round(1)
p('\n' + tab.round(2).to_string())
r, pv = stats.spearmanr(y.n_apps, y.n_ii)
p(f'\n  Spearman corr(applications, invites) = {r:+.3f}, p={pv:.5f}')
rho_yield, pv2 = stats.spearmanr(y.n_apps, y.yield_calc)
p(f'  Spearman corr(applications, YIELD)   = {rho_yield:+.3f}, p={pv2:.2e}   <-- the important one')
p('  More applications buys a LOWER hit rate. The marginal application is going to a program')
p('  that fits you worse, and the invite count barely moves.')
EXPORT['yield_bands'] = tab.reset_index().astype(object).where(pd.notna(tab.reset_index()), None).to_dict('records')

p('\n--- 21a. Marginal return: invites gained per 10 extra applications ---')
lo = y[y.n_apps <= 60]; mid = y[(y.n_apps > 60) & (y.n_apps <= 120)]; hi2 = y[y.n_apps > 120]
for nm, g in [('<=60 apps', lo), ('61-120 apps', mid), ('>120 apps', hi2)]:
    if len(g) < 10: continue
    sl = np.polyfit(g.n_apps, g.n_ii, 1)[0]
    p(f'  {nm:12s} n={len(g):3d}  slope = {10*sl:+.2f} invites per 10 applications')

# ======================================================================
h('22. WHAT PREDICTS AN INTERVIEW  (applicant-level, 24-25)')
m = a.dropna(subset=['yield_calc','step2','deg']).copy()
m['is_md'] = (m.deg=='MD').astype(int); m['is_img'] = (m.deg=='IMG').astype(int)
m['aoa_i'] = m.aoa_b.fillna(False).astype(int)
m['honors_i'] = m.m3_surg_honors_b.fillna(False).astype(int)
m['pubs_c'] = np.log1p(m.n_pubs.fillna(m.n_pubs.median()))
m['q1'] = (m.quart==1).astype(int)
m['score_c'] = (m.step2-255)/10
m['logapps'] = np.log(m.n_apps)
X = sm.add_constant(m[['score_c','is_img','aoa_i','honors_i','pubs_c','q1','logapps']])
mod = sm.GLM(m.n_ii, X, family=sm.families.Poisson(), offset=None).fit(cov_type='HC1')
p('Poisson model of INTERVIEW COUNT (HC1 robust SEs), n=%d' % len(m))
res = pd.DataFrame({'IRR': np.exp(mod.params), 'lo': np.exp(mod.conf_int()[0]),
                    'hi': np.exp(mod.conf_int()[1]), 'p': mod.pvalues}).round(3)
p(res.to_string())
p('\n  IRR = multiplicative effect on expected number of interviews.')
p('  score_c is per 10 Step 2 points; pubs_c is per log-unit of publications;')
p('  logapps is per log-unit of applications sent.')
EXPORT['poisson'] = [{'term': k, **{c: float(res.loc[k, c]) for c in res.columns}} for k in res.index]

# ======================================================================
h('23. WHAT PREDICTS MATCHING  (the outcome nobody else in this sheet has)')
o = a[a.status.notna()].copy()
p(f'applicants reporting an outcome: {len(o)} of {len(a)} ({100*len(o)/len(a):.0f}%)')
p(o.status.value_counts().to_string())
p('\nCAUTION: outcome reporting is voluntary and post-hoc. People who match are more likely to')
p('return and update the sheet, but people who do not match also come back to ask for advice.')
p('Direction of bias is genuinely unclear, so read these as within-sample contrasts.\n')
o['matched_cat'] = (o.status=='Categorical').astype(int)
by = o.groupby('deg').agg(n=('matched_cat','size'), pct_categorical=('matched_cat','mean'))
by['pct_categorical'] = (100*by.pct_categorical).round(1)
p(by.to_string())
p('\nBy interview count — the single strongest gradient in either workbook:')
o2 = o[o.n_ii.notna()].copy()
o2['iiband'] = pd.cut(o2.n_ii, [-.1,0.5,2.5,5.5,9.5,50],
                      labels=['0','1-2','3-5','6-9','10+'])
t = o2.groupby('iiband').agg(n=('matched_cat','size'), pct_cat=('matched_cat','mean'))
t['pct_cat'] = (100*t.pct_cat).round(1)
p(t.to_string())
EXPORT['match_by_ii'] = t.reset_index().astype(object).where(pd.notna(t.reset_index()), None).to_dict('records')
lo3 = o2[o2.n_ii <= 5]; hi3 = o2[o2.n_ii >= 10]
if len(lo3) > 5 and len(hi3) > 5:
    z, pv = sm.stats.proportions_ztest([lo3.matched_cat.sum(), hi3.matched_cat.sum()],
                                       [len(lo3), len(hi3)])
    p(f'\n  <=5 invites: {100*lo3.matched_cat.mean():.1f}% categorical (n={len(lo3)})')
    p(f'  >=10 invites: {100*hi3.matched_cat.mean():.1f}% categorical (n={len(hi3)})')
    p(f'  z={z:.2f}, p={pv:.2e}')

p('\nLogistic model of categorical match:')
mm = o.dropna(subset=['step2','n_ii']).copy()
mm['score_c'] = (mm.step2-255)/10
mm['is_img'] = (mm.deg=='IMG').astype(int); mm['is_do'] = (mm.deg=='DO').astype(int)
mm['aoa_i'] = mm.aoa_b.fillna(False).astype(int)
mm['ii_c'] = mm.n_ii/5
X2 = sm.add_constant(mm[['ii_c','score_c','is_img','is_do','aoa_i']])
lm = sm.GLM(mm.matched_cat, X2, family=sm.families.Binomial()).fit(cov_type='HC1')
r2 = pd.DataFrame({'OR': np.exp(lm.params), 'lo': np.exp(lm.conf_int()[0]),
                   'hi': np.exp(lm.conf_int()[1]), 'p': lm.pvalues}).round(3)
p(r2.to_string())
p(f'  n={len(mm)}. ii_c is per 5 interviews.')
p('  Interview count dominates: OR 9.1 per 5 interviews.')
p('  Step 2 flips NEGATIVE once interview count is held fixed (OR 0.58, p=0.02).')
p('  Do not read that as "a higher score hurts". It is collider stratification: conditioning')
p('  on interview count compares a 270 who got 8 interviews against a 240 who got 8, and the')
p('  270 who only managed 8 has something else going wrong (or aimed at harder programs).')
p('  The causal path for score runs THROUGH interview count, which the Poisson model above')
p('  already measured at IRR 1.22 per 10 points.')
EXPORT['match_logit'] = [{'term': k, **{c: float(r2.loc[k, c]) for c in r2.columns}} for k in r2.index]

# ======================================================================
h('24. TWO-YEAR COMPARISON')
p('--- 24a. What each cycle actually recorded ---')
p(f'  2024-25 workbook: APPLICANT-level. {len(a)} applicants, outcomes, yields, signal conversion.')
p(f'                    No date-stamped invite log -> no timing analysis possible.')
p(f'  2025-26 workbook: INVITE-level. {recs.weight.sum():.0f} invites across {recs.pkey.nunique()} programs,')
p(f'                    with dates -> timing analysis possible. No applicant IDs, no outcomes.')
p('  The two are complementary, not comparable head-to-head on most fields.')

p('\n--- 24b. Directly comparable: Step 2 distribution ---')
s24 = a.step2.dropna()
sc25 = recs.dropna(subset=['step2_bucket']).copy()
v25 = np.repeat((sc25.step2_bucket+5).values, sc25.weight.astype(int).values)
p(f'  24-25 APPLICANTS  (n={len(s24)}): median {s24.median():.0f}, IQR {s24.quantile(.25):.0f}-{s24.quantile(.75):.0f}, mean {s24.mean():.1f}')
p(f'  25-26 INVITEES    (n={len(v25)}): median {np.median(v25):.0f}, IQR {np.percentile(v25,25):.0f}-{np.percentile(v25,75):.0f}, mean {v25.mean():.1f}')
p('  NOT the same population: one is everyone who applied, the other is people who got invited.')
p('  The ~5 point gap is the selection effect of the interview filter, not year-over-year drift.')
s24i = a[a.n_ii.fillna(0) > 0].step2.dropna()
p(f'\n  24-25 applicants WITH >=1 invite (n={len(s24i)}): median {s24i.median():.0f}, mean {s24i.mean():.1f}')
p(f'  -> like-for-like against 25-26 invitees ({np.median(v25):.0f}): '
  f'difference {np.median(v25)-s24i.median():+.0f} points')
EXPORT['step2_compare'] = {
    'applicants_2425': [float(x) for x in s24.tolist()],
    'invited_2425': [float(x) for x in s24i.tolist()],
    'invitees_2526': [float(x) for x in v25.tolist()],
}

p('\n--- 24c. Directly comparable: program desirability (RANK SPOTS both years) ---')


def canon(x):
    x = re.sub(r'\([^)]*\)', '', str(x)); x = re.sub(r'[^a-z0-9 ]', ' ', x.lower())
    x = re.sub(r'\b(program|surgery|general|hospital|hospitals|medical|center|centre|health|healthcare|'
               r'university|univ|school|of|the|college|medicine|system|gme|clinic)\b', ' ', x)
    return re.sub(r'\s+', ' ', x).strip()


def rankspots(df):
    d = df.iloc[1:].copy()
    d.columns = ['program','city','state','total'] + [f'r{i}' for i in range(1,21)] + ['dnr']
    for c in [f'r{i}' for i in range(1,21)]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.program.notna()]
    d['n_rank'] = d[[f'r{i}' for i in range(1,21)]].sum(axis=1)
    d['pkey'] = d.program.map(canon)
    return d.groupby('pkey').agg(n_rank=('n_rank','sum'), r1=('r1','sum'),
                                 name=('program','first'), state=('state','first'))


r24 = rankspots(store24['RANK SPOTS']); r25 = rankspots(store25['RANK SPOTS 25-26'])
p(f'  24-25: {len(r24)} programs, {r24.n_rank.sum():.0f} rank slots, {r24.r1.sum():.0f} #1 ranks')
p(f'  25-26: {len(r25)} programs, {r25.n_rank.sum():.0f} rank slots, {r25.r1.sum():.0f} #1 ranks')
j = r24[['n_rank','r1','name','state']].join(r25[['n_rank','r1']], how='inner', lsuffix='_24', rsuffix='_25')
p(f'  programs present in both: {len(j)}')
rho, pv = stats.spearmanr(j.n_rank_24, j.n_rank_25)
p(f'  Spearman corr of demand across years = {rho:+.3f} (p={pv:.2e})')
p('  Moderately stable — the broad ordering persists, but see the top-25 churn below before')
p('  treating any single programme\'s movement as real.')
p('\n  Biggest movers (rank-slot count). READ WITH CARE: rank slots are small counts driven by')
p('  who happened to fill in the sheet, so most of this is reporting noise, not demand shift.')
p('  UCLA going 30 -> 6 is far more likely a collapse in respondents than a collapse in demand.')
j['delta'] = j.n_rank_25 - j.n_rank_24
p('  RISERS:'); p(j.nlargest(8,'delta')[['name','state','n_rank_24','n_rank_25','delta']].to_string(index=False, max_colwidth=44))
p('  FALLERS:'); p(j.nsmallest(8,'delta')[['name','state','n_rank_24','n_rank_25','delta']].to_string(index=False, max_colwidth=44))
top24 = set(r24.nlargest(25,'n_rank').index); top25 = set(r25.nlargest(25,'n_rank').index)
p(f'\n  Overlap of the top-25 most-ranked programs across years: {len(top24&top25)}/25')
EXPORT['demand_scatter'] = [{'name': str(rr['name'])[:46], 'state': str(rr.state),
                             'y24': float(rr.n_rank_24), 'y25': float(rr.n_rank_25)}
                            for _, rr in j.iterrows()]

p('\n--- 24d. Gini of demand, both years ---')
for lab, d in [('24-25', r24.n_rank), ('25-26', r25.n_rank)]:
    v = np.sort(d.values); tot = v.sum()
    g = 1 - 2*np.trapezoid(np.cumsum(v)/tot, dx=1/len(v))
    p(f'  {lab}: Gini {g:.3f} | top-25 programs hold {100*d.nlargest(25).sum()/tot:.1f}% of demand')

# ======================================================================
h('25. HEADLINE NUMBERS FOR THE SITE')
E = {
 'applicants_2425': int(len(a)),
 'invites_2526': int(recs.weight.sum()),
 'programs_2526': int(recs.pkey.nunique()),
 'sig_mean': round(float(s.sig_converted.mean()), 2),
 'sig_pct': round(100*float(s.sig_converted.mean())/15, 1),
 'sig_zero_pct': round(100*float((s.sig_converted==0).mean()), 1),
 'median_apps': int(y.n_apps.median()),
 'median_ii': int(y.n_ii.median()),
 'median_yield': round(100*float(y.yield_calc.median()), 1),
 'yield_vs_apps_rho': round(float(rho_yield), 3),
 'release_day_share': 35.4,
 'signal_or': 3.85,
 'tier_gap_pp': 24.8,
 'demand_stability_rho': round(float(rho), 3),
}
for k, v in E.items(): p(f'  {k:24s} {v}')
EXPORT['headline'] = E
json.dump(EXPORT, open('site_data.json','w'), indent=1, default=str)
open('analysis_part4.txt','w').write('\n'.join(OUT))
a.to_pickle('appl24_clean.pkl')
p('\n[written analysis_part4.txt, site_data.json]')
