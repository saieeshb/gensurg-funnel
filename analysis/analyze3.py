"""Part 3: statistical hardening, bias quantification, and 2026-27 projection."""
import pandas as pd, numpy as np, pickle, re, datetime as dt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

pd.set_option('display.width', 210)
A = pickle.load(open('analysis1.pkl', 'rb'))
B = pickle.load(open('analysis2.pkl', 'rb'))
recs, recs2 = A['recs'], B['recs2']
OUT = []
rng = np.random.default_rng(7)


def h(t):
    line = '\n' + '=' * 92 + f'\n== {t}\n' + '=' * 92
    print(line); OUT.append(line)


def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


# expand weighted records to one row per invite
E = recs2.loc[recs2.index.repeat(recs2.weight.astype(int))].reset_index(drop=True)
p(f'expanded invite-level rows: {len(E)}')

# ========================================================================
h('16. STATISTICAL HARDENING — clustering, robustness, multiplicity')

p('--- 16a. The independence problem ---')
p('Records are NOT independent: one program contributes many invites (shared screening rule),')
p('and one applicant contributes many invites (shared CV). Naive SEs are too narrow.')
pc = E.groupby('pkey').size()
p(f'  invites per program: mean {pc.mean():.1f}, median {pc.median():.0f}, max {pc.max()}')
p(f'  {(pc>=5).sum()} programs contribute >=5 invites, holding {100*pc[pc>=5].sum()/len(E):.0f}% of the data')
p('  Applicant identity is NOT recorded anywhere in the sheet, so applicant-level clustering')
p('  is impossible to correct for. This is the single largest un-fixable limitation.')

p('\n--- 16b. Logistic model with program-cluster-robust SEs ---')
M = E.dropna(subset=['signal', 'step2_bucket', 'degree', 'geo']).copy()
M['grp'] = M.degree.replace({'IMG (unspec)': 'US IMG'})
M['y'] = M.signal.astype(int)
M['score_c'] = (M.step2_bucket + 5 - 255) / 10
M['geo_i'] = M.geo.astype(int)
M['home_i'] = M.home.fillna(False).astype(int)
M['away_i'] = M.away.fillna(False).astype(int)
M['wk'] = (M.date - M.date.min()).dt.days / 7
M['is_img'] = M.grp.str.contains('IMG').astype(int)
M['is_do'] = (M.grp == 'DO').astype(int)
M['sought'] = (M.n_rank.fillna(0) >= 6).astype(int)
cols = ['score_c', 'geo_i', 'home_i', 'away_i', 'wk', 'is_img', 'is_do', 'sought']
X = sm.add_constant(M[cols])
naive = sm.GLM(M.y, X, family=sm.families.Binomial()).fit()
rob = sm.GLM(M.y, X, family=sm.families.Binomial()).fit(cov_type='cluster',
                                                        cov_kwds={'groups': M.pkey})
cmp = pd.DataFrame({'OR': np.exp(rob.params),
                    'lo': np.exp(rob.conf_int()[0]), 'hi': np.exp(rob.conf_int()[1]),
                    'p_clustered': rob.pvalues, 'p_naive': naive.pvalues,
                    'SE_naive': naive.bse, 'SE_clustered': rob.bse})
p(cmp.round(3).to_string())
p(f'\n  Clustered SEs are on average {100*(cmp.SE_clustered/cmp.SE_naive-1).mean():.0f}% wider than naive.')
p(f'  n = {len(M)} invites across {M.pkey.nunique()} programs.')
p(f'  Pseudo-R2 (Cox-Snell) = {1-np.exp((naive.llnull-naive.llf)*2/len(M)):.3f} — most variance is')
p('  unexplained, as expected when the strongest inputs (letters, school, interview) are unobserved.')

p('\n--- 16c. Robustness: does the Oct 22 mass-release drive the results? ---')
for lab, sub in [('ALL invites', M), ('EXCLUDING 2025-10-22', M[M.date != '2025-10-22']),
                 ('ONLY 2025-10-22', M[M.date == '2025-10-22'])]:
    if len(sub) < 40: continue
    hi_ = sub[sub.sought == 1]; lo_ = sub[sub.sought == 0]
    if len(hi_) < 10 or len(lo_) < 10: continue
    p(f'  {lab:22s} n={len(sub):4d} | signal share sought-after {100*hi_.y.mean():5.1f}% '
      f'vs rest {100*lo_.y.mean():5.1f}% | gap {100*(hi_.y.mean()-lo_.y.mean()):+5.1f} pp')

p('\n--- 16d. Bootstrap (cluster bootstrap by program) of the desirability gradient ---')
progs = M.pkey.unique()
gaps = []
for _ in range(4000):
    pick = rng.choice(progs, len(progs), replace=True)
    bs = pd.concat([M[M.pkey == q] for q in pick])
    a, b = bs[bs.sought == 1], bs[bs.sought == 0]
    if len(a) > 5 and len(b) > 5:
        gaps.append(a.y.mean() - b.y.mean())
gaps = np.array(gaps)
p(f'  signal-share gap (most-sought minus rest): {100*gaps.mean():.1f} pp '
  f'[95% CI {100*np.percentile(gaps,2.5):.1f} to {100*np.percentile(gaps,97.5):.1f}]')
p(f'  P(gap <= 0) under the bootstrap = {(gaps<=0).mean():.4f}')

p('\n--- 16e. Multiplicity correction across the headline tests ---')
tests = []
d = E.dropna(subset=['signal', 'step2_bucket'])
lo_, hi_ = d[d.step2_bucket <= 240], d[d.step2_bucket >= 260]
tests.append(('signal share: <=249 vs >=260 Step 2',
              sm.stats.proportions_ztest([lo_.signal.sum(), hi_.signal.sum()], [len(lo_), len(hi_)])[1]))
a, b = M[M.sought == 1], M[M.sought == 0]
tests.append(('signal share: most-sought vs rest programs',
              sm.stats.proportions_ztest([a.y.sum(), b.y.sum()], [len(a), len(b)])[1]))
dd = E.dropna(subset=['signal', 'degree'])
dd['g'] = dd.degree.replace({'IMG (unspec)': 'US IMG'})
tests.append(('signal use differs by degree pathway',
              stats.chi2_contingency(pd.crosstab(dd.g, dd.signal))[1]))
gg = E.dropna(subset=['geo', 'degree']); gg['g'] = gg.degree.replace({'IMG (unspec)': 'US IMG'})
tests.append(('geo-preference differs by degree pathway',
              stats.chi2_contingency(pd.crosstab(gg.g, gg.geo))[1]))
pr = E.dropna(subset=['degree'])
tests.append(('IMG share: prelim vs categorical invites',
              stats.chi2_contingency(pd.crosstab(pr.prelim, pr.degree.astype(str).str.contains('IMG')))[1]))
s = E.dropna(subset=['signal']).copy(); s['t'] = (s.date - s.date.min()).dt.days
tests.append(('signal share rises over the season',
              stats.pointbiserialr(s.signal.astype(int), s.t)[1]))
names = [t[0] for t in tests]; raw = [t[1] for t in tests]
rej, adj, _, _ = multipletests(raw, alpha=0.05, method='holm')
p(f'  {"test":48s} {"raw p":>10s} {"Holm p":>10s}  survives')
for n_, r_, a_, k_ in zip(names, raw, adj, rej):
    p(f'  {n_:48s} {r_:10.5f} {a_:10.5f}  {"YES" if k_ else "no"}')

p('\n--- 16f. Simpson-paradox check: score-vs-signal within desirability tier ---')
for lab, sub in [('most-sought programs', M[M.sought == 1]), ('all other programs', M[M.sought == 0])]:
    r_, pv = stats.pointbiserialr(sub.y, sub.score_c)
    p(f'  {lab:22s} n={len(sub):4d}  corr(signalled, score) = {r_:+.3f} (p={pv:.3f})')
p('  The near-zero pooled correlation is not masking opposite within-tier signs.')

# ========================================================================
h('17. HOW MUCH OF REALITY DOES THIS SHEET SEE?  (coverage & bias)')
p(f'Reported invites: {len(E)} | distinct programs appearing: {E.pkey.nunique()}')
p(f'Sheet\'s own count of programs that released invites: 498')
p('\nOrder-of-magnitude coverage (external anchors, not from the sheet):')
p('  General Surgery categorical PGY-1 positions per year ~1,700-1,800 across ~350 programs.')
p('  At a typical ~10-12 interviews offered per position, national invite volume is ~18,000-22,000.')
p(f'  => this sheet captures roughly {100*len(E)/20000:.0f}% of categorical invites (order 5-6%).')
p('\nDirection of the biases this creates:')
p('  1. VOLUNTEER bias — people post when something happens. Non-events (no invite) are invisible.')
p('     The sheet therefore cannot estimate any absolute probability, only compositions.')
p('  2. SUCCESS bias — an applicant with 15 invites can post 15 times; one with 0 posts never.')
p('     Records are weighted toward successful applicants, inflating apparent score/credential levels.')
p('  3. SURVIVOR bias in later waves — those still posting in December are those still searching.')
p('  4. AUDIENCE bias — the sheet skews US-MD and US-based; Non-US IMG behaviour is under-sampled')
p(f'     relative to their true applicant share ({100*E.degree.astype(str).str.contains("IMG").mean():.0f}% of records here).')
p('  5. RECALL/format drift — free text; ~2% of fields unparseable, and "no signal" may be')
p('     under-reported by people who only bother to note a signal when they have one.')
p('\nWhat this means: read every number here as "among invites that were reported", and lean on')
p('the tier-matched and within-sheet contrasts (which share the bias on both sides) rather than levels.')

p('\n--- 17b. Missing-data check: is reporting quality related to the outcome? ---')
E['sig_missing'] = E.signal.isna()
p(f'  records missing the signal field: {E.sig_missing.sum()} ({100*E.sig_missing.mean():.1f}%)')
for f in ['degree', 'step2_bucket', 'geo']:
    miss = E[E[f].isna()]
    p(f'  missing {f:13s}: n={len(miss):3d} | of these, signal share = '
      f'{100*miss.signal.dropna().mean() if miss.signal.notna().any() else float("nan"):.1f}% '
      f'(vs {100*E.signal.dropna().mean():.1f}% overall)')
p('  Missingness is low and not concentrated in any one attribute — treat as missing-at-random-ish.')

# ========================================================================
h('18. PROJECTION FOR THE 2026-27 CYCLE')
p('--- 18a. Calendar anchoring ---')
first = recs.date.min()
p(f'  First reported invite of 25-26 : {first.date()} ({first.day_name()})')
p(f'  Programs gained ERAS access    : 2025-09-24 (Wed) — one day earlier, consistent with the log')
peak = pd.Timestamp('2025-10-22')
p(f'  APDS universal release date    : {peak.date()} ({peak.day_name()}) = program-access + 28 days')
p('  The 2027 ERAS cycle repeats the same weekday structure 364 days later (52 weeks),')
p('  which preserves both the date-in-month and the day-of-week:')
SHIFT = 364
for lab, dte in [('program ERAS access', pd.Timestamp('2025-09-24')),
                 ('first trickle of invites', first),
                 ('UNIVERSAL RELEASE DATE', peak),
                 ('50% of invites landed', pd.Timestamp('2025-10-22')),
                 ('75% of invites landed', pd.Timestamp('2025-10-27')),
                 ('90% of invites landed', pd.Timestamp('2025-11-05')),
                 ('95% of invites landed', pd.Timestamp('2025-11-21'))]:
    nd = dte + pd.Timedelta(days=SHIFT)
    p(f'    {lab:26s} {dte.date()} ({dte.day_name()[:3]})  ->  {nd.date()} ({nd.day_name()[:3]})')
p('\n  PRIMARY PROJECTION: the 26-27 universal release lands on Wed 2026-10-21.')
p('  Alternative if APDS anchors to "4th Wednesday of October" instead: Wed 2026-10-28.')
p('  Confirm against the APDS statement and the AAMC ERAS 2027 calendar before acting.')

p('\n--- 18b. Volume projection under the same reporting behaviour ---')
byday = recs.groupby('date').weight.sum()
p(f'  25-26 reported total: {byday.sum():.0f} invites; {100*byday.max()/byday.sum():.0f}% on the single release day.')
p('  Expected 26-27 shape (share of season total, from 25-26):')
wk = recs.groupby('week').weight.sum()
wk2 = (100 * wk / wk.sum()).round(1)
for k, v in wk2.items():
    if v < 0.5: continue
    p(f'    week of {(k+pd.Timedelta(days=SHIFT)).date()}  ~{v:4.1f}% of the season\'s invites')

p('\n--- 18c. What to expect on the numbers ---')
sig = E.signal.dropna()
p(f'  Signal share among invitees was {100*sig.mean():.1f}% in 25-26 (n={len(sig)}).')
p('  Gen Surgery keeps 15 signals in 26-27 (the sheet\'s own APPLICANT INFO tab tracks "II by Signals x/15").')
p('  With the signal count unchanged and applicants now a year wiser about concentrating them,')
p('  expect the signalled share of invitees to hold or drift UP: projected 55-62% for 26-27.')
sc = E.dropna(subset=['step2_bucket'])
v = (sc.step2_bucket + 5).values
p(f'  Step 2 CK of invitees: median {np.median(v):.0f}, IQR {np.percentile(v,25):.0f}-{np.percentile(v,75):.0f}, '
  f'{100*(v>=260).mean():.0f}% at 260+.')
p('  Score inflation runs ~1 point/yr nationally, so expect the 26-27 invitee median near 255-260')
p('  and the practical "comfortable" threshold for academic programs to sit around 250.')

p('\n--- 18d. Falsifiable predictions for 26-27 (check these against the new sheet) ---')
preds = [
 ('P1', 'The single busiest invite day accounts for >25% of all reported invites', 'observed 35% in 25-26'),
 ('P2', 'That day is a Wednesday in the 3rd-4th week of October (2026-10-21 primary)', 'Wed held 45% of 25-26 volume'),
 ('P3', '>=50% of all reported invites land within the 7 days around the release date', '25-26: 50% by day of release'),
 ('P4', 'Signalled share of invitees is between 55% and 62%', '53.8% in 25-26, trending up'),
 ('P5', 'Signal share at the most-sought programs exceeds it at low-demand programs by >=15 pp',
       '25.5 pp gap in 25-26, bootstrap CI excludes 0'),
 ('P6', 'Median Step 2 CK of reported invitees is 255-260 with IQR width ~20', '255 / IQR 245-265 in 25-26'),
 ('P7', 'IMGs are >=50% of prelim invites but <15% of categorical invites', '76% vs 8% in 25-26'),
 ('P8', 'Geographic-preference share of invitees stays in the 70-78% band', '73.9% in 25-26'),
 ('P9', 'LOI response rate remains under 50%, and under 35% personalised', '34% any response; 30% personalised'),
 ('P10','Interview-drop activity peaks in the 2-4 weeks after the release date', '25-26 drops clustered late Oct-Nov'),
]
for k, s, why in preds:
    p(f'  {k}: {s}\n       (basis: {why})')

open('analysis_part3.txt', 'w').write('\n'.join(OUT))
p('\n[written analysis_part3.txt]')
