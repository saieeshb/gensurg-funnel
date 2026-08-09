"""Advanced analysis of the 2026-2027 Gen Surgery spreadsheet (25-26 cycle data)."""
import pandas as pd, numpy as np, pickle, re, datetime as dt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.width', 200)
store = pickle.load(open('sheets.pkl', 'rb'))
recs = pd.read_pickle('recs.pkl')
OUT = []


def h(t):
    line = '\n' + '=' * 92 + f'\n== {t}\n' + '=' * 92
    print(line); OUT.append(line)


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s); OUT.append(s)


def canon(s):
    s = re.sub(r'\([^)]*\)', '', str(s))
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    s = re.sub(r'\b(program|surgery|general|hospital|hospitals|medical|center|centre|health|healthcare|'
               r'university|univ|school|of|the|college|medicine|system|gme|clinic)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


recs['pkey'] = recs.program_raw.map(canon)
recs['week'] = recs.date.dt.to_period('W').dt.start_time
recs['dow'] = recs.date.dt.day_name()
W = recs.weight

# =========================================================================
h('1. DATASET INVENTORY')
p(f'Weighted invite records parsed from "II by Date 25-26": {W.sum():.0f}')
p(f'Distinct reporting rows: {len(recs)} | distinct programs (canonicalised): {recs.pkey.nunique()}')
p(f'Date span: {recs.date.min().date()} to {recs.date.max().date()}  '
  f'({(recs.date.max()-recs.date.min()).days} days)')
ii = store['II by Date 25-26']
p(f'Sheet\'s own header count of programs that released invites: {ii.iat[3,0]}')
rel = pd.DataFrame({'date': [ii.iat[r,0] for r in range(ii.shape[0])],
                    'nprog': [ii.iat[r,1] for r in range(ii.shape[0])]})
rel = rel[rel.date.apply(lambda v: isinstance(v,(dt.datetime,pd.Timestamp)))]
rel['nprog'] = pd.to_numeric(rel.nprog, errors='coerce')
rel['date'] = pd.to_datetime(rel.date)
p(f'Days logged with a release count: {rel.nprog.notna().sum()} | sum of daily program releases: {rel.nprog.sum():.0f}')

# =========================================================================
h('2. WHO GETS INTERVIEWED — RAW COMPOSITION OF 1,154 INVITES')
for f in ['degree']:
    t = recs.groupby(f).weight.sum().sort_values(ascending=False)
    p(f'\n{f}:'); p((100*t/t.sum()).round(1).to_string(), f'\n  total={t.sum():.0f}')

p('\nStep 2 CK (decade bucket):')
sb = recs.groupby('step2_bucket').weight.sum()
cum = 100*sb.cumsum()/sb.sum()
for k in sb.index:
    p(f'  {int(k)}s : {sb[k]:5.0f}  {100*sb[k]/sb.sum():5.1f}%   cum {cum[k]:5.1f}%')
# weighted mean/median of bucket midpoints
mid = recs.dropna(subset=['step2_bucket']).copy(); mid['m'] = mid.step2_bucket + 5
wm = np.average(mid.m, weights=mid.weight)
vals = np.repeat(mid.m.values, mid.weight.astype(int).values)
p(f'  weighted mean ~{wm:.1f} | median ~{np.median(vals):.0f} | IQR {np.percentile(vals,25):.0f}-{np.percentile(vals,75):.0f}')

p('\nBinary attributes (share of invitees with the attribute, among records where reported):')
for f in ['signal', 'geo', 'home', 'away']:
    s = recs[recs[f].notna()]
    k = s[s[f]].weight.sum(); n = s.weight.sum()
    lo, hi = stats.beta.ppf([.025,.975], k+.5, n-k+.5)
    p(f'  {f:7s}: {k:5.0f}/{n:.0f} = {100*k/n:5.1f}%  [95% CI {100*lo:.1f}-{100*hi:.1f}]')

# =========================================================================
h('3. BASE RATES AND SIGNAL LIFT  (the denominator problem)')
c = store['Comm 25-26'].iloc[2:, [0,1,2,3,4,6,7]].copy()
c.columns = ['program','date','state','applied','recv','applied_nosig','recv_nosig']
for col in ['applied','recv','applied_nosig','recv_nosig']:
    c[col] = pd.to_numeric(c[col], errors='coerce')
c = c.dropna(subset=['applied','applied_nosig'])
c['applied_sig'] = c.applied - c.applied_nosig
c['pkey'] = c.program.map(canon)
tot_app, tot_nosig = c.applied.sum(), c.applied_nosig.sum()
base = (tot_app - tot_nosig) / tot_app
p(f'From "Comm 25-26" (per-program tallies of sheet users who applied, split by signal):')
p(f'  {len(c)} programs | {tot_app:.0f} applications | {tot_app-tot_nosig:.0f} carried a signal')
p(f'  => P(signal | application to THIS class of program) = {100*base:.1f}%')
p(f'  NOTE: these are mostly large academic programs that solicit supplemental apps,')
p(f'        so this is a tier-specific, upward-biased estimate of the global signal rate.')

p('\n--- 3a. TIER-MATCHED comparison (same programs, both sheets) ---')
inv = recs[recs.signal.notna()].groupby('pkey').apply(
    lambda d: pd.Series({'inv_n': d.weight.sum(), 'inv_sig': d[d.signal].weight.sum()}), include_groups=False)
m = c.groupby('pkey')[['applied','applied_sig']].sum().join(inv, how='inner')
m = m[m.inv_n >= 1]
p(m.to_string())
A, As = m.applied.sum(), m.applied_sig.sum()
I, Is = m.inv_n.sum(), m.inv_sig.sum()
p(f'\n  Applications to these programs : {A:.0f}, signalled {As:.0f} ({100*As/A:.1f}%)')
p(f'  Invites from these programs    : {I:.0f}, signalled {Is:.0f} ({100*Is/I:.1f}%)')
odds_app = As/(A-As); odds_inv = Is/(I-Is)
OR = odds_inv/odds_app
se = np.sqrt(1/As + 1/(A-As) + 1/Is + 1/(I-Is))
p(f'  Odds of having signalled: applicants {odds_app:.3f} -> invitees {odds_inv:.3f}')
p(f'  ==> SIGNAL ODDS RATIO = {OR:.2f}  [95% CI {np.exp(np.log(OR)-1.96*se):.2f}-{np.exp(np.log(OR)+1.96*se):.2f}]')
p(f'  ==> Relative risk of interview, signal vs no signal ~ {(Is/As)/((I-Is)/(A-As)):.2f}x')

p('\n--- 3b. Sensitivity: global signal lift vs assumed applications per applicant ---')
p('  (15 signals available in Gen Surg; base rate = 15/N_apps)')
p(f'  {"N apps":>7} | {"base rate":>9} | {"odds ratio":>10} | {"interpretation"}')
sig_share = recs[recs.signal.notna()]
obs = sig_share[sig_share.signal].weight.sum()/sig_share.weight.sum()
for n_apps in [30, 40, 50, 60, 70, 80, 100]:
    b = 15/n_apps
    or_ = (obs/(1-obs))/(b/(1-b))
    p(f'  {n_apps:7d} | {100*b:8.1f}% | {or_:10.2f} | signal multiplies interview odds ~{or_:.1f}x')
p(f'\n  Observed signal share among ALL invitees = {100*obs:.1f}%')

# =========================================================================
h('4. DOES A SIGNAL SUBSTITUTE FOR A SCORE?  (compensation analysis)')
p('Logic: among invitees, if low-score applicants are disproportionately signallers,')
p('the signal is doing work the score is not. (Bayes: signal-share among invitees')
p('tracks the selection lift when applicants\' signalling propensity is score-independent.)\n')
d = recs.dropna(subset=['signal','step2_bucket']).copy()
tab = d.groupby('step2_bucket').apply(
    lambda x: pd.Series({'n': x.weight.sum(), 'sig': x[x.signal].weight.sum()}), include_groups=False)
tab['pct_signalled'] = 100*tab.sig/tab.n
p(tab.round(1).to_string())
lo = d[d.step2_bucket <= 240]; hi = d[d.step2_bucket >= 260]
a1, n1 = lo[lo.signal].weight.sum(), lo.weight.sum()
a2, n2 = hi[hi.signal].weight.sum(), hi.weight.sum()
p(f'\n  <=249 invitees signalled: {a1:.0f}/{n1:.0f} = {100*a1/n1:.1f}%')
p(f'  >=260 invitees signalled: {a2:.0f}/{n2:.0f} = {100*a2/n2:.1f}%')
z, pv = sm.stats.proportions_ztest([a1,a2],[n1,n2])
p(f'  difference = {100*(a1/n1-a2/n2):+.1f} pp, z={z:.2f}, p={pv:.4f}')
# trend test
x = np.repeat(d.step2_bucket.values, d.weight.astype(int).values)
y = np.repeat(d.signal.values.astype(int), d.weight.astype(int).values)
r, pr = stats.pointbiserialr(y, x)
p(f'  point-biserial corr(signal, score) = {r:+.3f} (p={pr:.4f})')

p('\n--- 4b. Same test for geographic preference ---')
d2 = recs.dropna(subset=['geo','step2_bucket']).copy()
tab2 = d2.groupby('step2_bucket').apply(
    lambda x: pd.Series({'n': x.weight.sum(), 'geo': x[x.geo].weight.sum()}), include_groups=False)
tab2['pct_geo'] = 100*tab2.geo/tab2.n
p(tab2.round(1).to_string())

# =========================================================================
h('5. DEGREE-STRATIFIED REQUIREMENTS  (what each pathway needs to bring)')
d = recs.dropna(subset=['degree']).copy()
d['grp'] = d.degree.replace({'IMG (unspec)':'US IMG'})
rows = []
for g, x in d.groupby('grp'):
    r = {'group': g, 'n_invites': x.weight.sum()}
    for f in ['signal','geo','home','away']:
        s = x[x[f].notna()]
        r[f'%{f}'] = 100*s[s[f]].weight.sum()/s.weight.sum() if s.weight.sum() else np.nan
    sc = x.dropna(subset=['step2_bucket'])
    v = np.repeat((sc.step2_bucket+5).values, sc.weight.astype(int).values)
    r['median_step2'] = np.median(v); r['p25'] = np.percentile(v,25); r['p75'] = np.percentile(v,75)
    r['%>=260'] = 100*sc[sc.step2_bucket>=260].weight.sum()/sc.weight.sum()
    rows.append(r)
p(pd.DataFrame(rows).set_index('group').round(1).to_string())

p('\nChi-square: signal use differs by degree pathway?')
ct = pd.crosstab(d[d.signal.notna()].grp, d[d.signal.notna()].signal,
                 values=d[d.signal.notna()].weight, aggfunc='sum').fillna(0)
chi2, pv, dof, _ = stats.chi2_contingency(ct)
p(ct.to_string()); p(f'  chi2={chi2:.2f}, dof={dof}, p={pv:.5f}')

# =========================================================================
h('6. MULTIVARIABLE MODEL — what predicts being a SIGNALLED invitee')
mdf = recs.dropna(subset=['signal','step2_bucket','degree','geo']).copy()
mdf['grp'] = mdf.degree.replace({'IMG (unspec)':'US IMG'})
mdf['sig'] = mdf.signal.astype(int)
mdf['score_c'] = (mdf.step2_bucket + 5 - 255)/10
mdf['geo_i'] = mdf.geo.astype(int)
mdf['home_i'] = mdf.home.fillna(False).astype(int)
mdf['away_i'] = mdf.away.fillna(False).astype(int)
mdf['wk'] = (mdf.date - mdf.date.min()).dt.days/7
mdf['is_md'] = (mdf.grp=='MD').astype(int)
mdf['is_img'] = mdf.grp.str.contains('IMG').astype(int)
X = mdf[['score_c','geo_i','home_i','away_i','wk','is_img']].copy()
X['is_do'] = (mdf.grp=='DO').astype(int)
X = sm.add_constant(X)
mod = sm.GLM(mdf.sig, X, family=sm.families.Binomial(), freq_weights=mdf.weight).fit()
p(mod.summary().as_text())
p('\nOdds ratios (reference = MD, week 0, score 255, no geo/home/away):')
ors = pd.DataFrame({'OR': np.exp(mod.params), 'lo': np.exp(mod.conf_int()[0]),
                    'hi': np.exp(mod.conf_int()[1]), 'p': mod.pvalues}).round(3)
p(ors.to_string())

# =========================================================================
h('7. TIMING — the shape of interview season')
byday = recs.groupby('date').weight.sum()
p('Weekly invite volume (reported):')
wk = recs.groupby('week').weight.sum()
for k, v in wk.items():
    p(f'  {k.date()}  {int(v):4d}  {"#"*int(v/4)}')
cs = byday.cumsum()/byday.sum()
for q in [.10,.25,.50,.75,.90,.95]:
    dte = cs[cs>=q].index[0]
    p(f'  {int(q*100)}% of all reported invites had landed by {dte.date()}')
p('\nBusiest single days:')
p(byday.sort_values(ascending=False).head(10).to_string())
p('\nBy day of week (weighted):')
dw = recs.groupby('dow').weight.sum().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
for k,v in dw.items():
    p(f'  {k:10s} {int(v):4d}  {100*v/dw.sum():5.1f}%  {"#"*int(v/6)}')

p('\nSignal share of invitees over time (are signalled invites front-loaded?):')
tw = recs[recs.signal.notna()].groupby('week').apply(
    lambda x: pd.Series({'n': x.weight.sum(), 'sig': x[x.signal].weight.sum()}), include_groups=False)
tw['pct'] = (100*tw.sig/tw.n).round(1)
p(tw[tw.n>=15].to_string())
sub = recs[recs.signal.notna()].copy()
sub['t'] = (sub.date - sub.date.min()).dt.days
xx = np.repeat(sub.t.values, sub.weight.astype(int).values)
yy = np.repeat(sub.signal.values.astype(int), sub.weight.astype(int).values)
r, pr = stats.pointbiserialr(yy, xx)
p(f'  corr(signalled, days since first invite) = {r:+.3f}, p={pr:.4f}')

p('\nMedian Step 2 of invitees over time (score drift):')
sc = recs.dropna(subset=['step2_bucket']).copy()
for k, x in sc.groupby('week'):
    if x.weight.sum() < 15: continue
    v = np.repeat((x.step2_bucket+5).values, x.weight.astype(int).values)
    p(f'  {k.date()}  n={int(x.weight.sum()):4d}  median {np.median(v):.0f}  mean {v.mean():.1f}')

# =========================================================================
h('8. THE "OCTOBER 22" MASS-RELEASE PHENOMENON')
top = byday.sort_values(ascending=False).head(3)
for dte in top.index:
    x = recs[recs.date==dte]
    npro = x.pkey.nunique()
    s = x[x.signal.notna()]
    p(f'{dte.date()}: {int(x.weight.sum())} invites across {npro} programs | '
      f'signal share {100*s[s.signal].weight.sum()/s.weight.sum():.1f}%')
p(f'\nShare of the whole season\'s reported invites on the single biggest day: '
  f'{100*top.iloc[0]/byday.sum():.1f}%')
p(f'Share in the 3 biggest days: {100*top.sum()/byday.sum():.1f}%')

pickle.dump({'recs':recs,'comm':c,'matched':m,'byday':byday,'rel':rel}, open('analysis1.pkl','wb'))
open('analysis_part1.txt','w').write('\n'.join(OUT))
p('\n[written analysis_part1.txt]')
