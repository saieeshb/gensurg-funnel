"""Part 2: program tiers, geography, IMG pathway, post-interview signals, churn."""
import pandas as pd, numpy as np, pickle, re, datetime as dt
from scipy import stats
import statsmodels.api as sm

pd.set_option('display.width', 210)
store = pickle.load(open('sheets.pkl', 'rb'))
A = pickle.load(open('analysis1.pkl', 'rb'))
recs = A['recs']
OUT = []


def h(t):
    line = '\n' + '=' * 92 + f'\n== {t}\n' + '=' * 92
    print(line); OUT.append(line)


def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def canon(s):
    s = re.sub(r'\([^)]*\)', '', str(s))
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    s = re.sub(r'\b(program|surgery|general|hospital|hospitals|medical|center|centre|health|healthcare|'
               r'university|univ|school|of|the|college|medicine|system|gme|clinic)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# ========================================================================
h('9. PROGRAM DESIRABILITY (from "RANK SPOTS 25-26") AND WHO IT INVITES')
rs = store['RANK SPOTS 25-26'].iloc[1:].copy()
rs.columns = ['program', 'city', 'state', 'total'] + [f'r{i}' for i in range(1, 21)] + ['dnr']
rs['total'] = pd.to_numeric(rs.total, errors='coerce')
for c in [f'r{i}' for i in range(1, 21)]:
    rs[c] = pd.to_numeric(rs[c], errors='coerce')
rs = rs.dropna(subset=['program'])
rs['n_rank'] = rs[[f'r{i}' for i in range(1, 21)]].sum(axis=1)
rs['n_top3'] = rs[['r1', 'r2', 'r3']].sum(axis=1)
rs['pkey'] = rs.program.map(canon)
p(f'programs listed: {len(rs)} | programs with >=1 rank: {(rs.n_rank>0).sum()} | total rank slots: {rs.n_rank.sum():.0f}')
p(f'Of {rs.n_rank.sum():.0f} rank-list slots reported, {rs.r1.sum():.0f} are #1 ranks '
  f'=> implied median rank-list depth ~{rs.n_rank.sum()/rs.r1.sum():.1f} programs per applicant')
p('\nMost-desired programs (by #1 ranks, then top-3):')
p(rs.nlargest(18, ['r1', 'n_top3'])[['program', 'state', 'n_rank', 'r1', 'n_top3']]
  .to_string(index=False, max_colwidth=48))

p('\nConcentration of demand:')
nr = rs.n_rank.sort_values(ascending=False)
tot = nr.sum()
for k in [10, 25, 50, 100]:
    p(f'  top {k:3d} programs hold {100*nr.head(k).sum()/tot:5.1f}% of all reported rank slots')
gini = 1 - 2*np.trapezoid(np.cumsum(np.sort(nr.values))/tot, dx=1/len(nr))
p(f'  Gini coefficient of rank-slot demand across {len(nr)} programs = {gini:.3f}')

p('\n--- 9b. Invite composition by program desirability tier ---')
inv = recs.groupby('pkey').weight.sum().rename('inv_n')
j = rs.groupby('pkey').agg(n_rank=('n_rank','sum'), r1=('r1','sum')).join(inv, how='inner')
j = j[j.inv_n >= 1]
p(f'programs matched between RANK SPOTS and invite log: {len(j)}')
recs2 = recs.merge(j[['n_rank','r1']], left_on='pkey', right_index=True, how='left')
recs2['tier'] = pd.cut(recs2.n_rank, [-.1,.5,2.5,5.5,100],
                       labels=['unranked by anyone','1-2 ranks','3-5 ranks','6+ ranks (most sought)'])
rows=[]
for t, x in recs2.dropna(subset=['tier']).groupby('tier', observed=True):
    s = x[x.signal.notna()]; g = x[x.geo.notna()]; sc = x.dropna(subset=['step2_bucket'])
    v = np.repeat((sc.step2_bucket+5).values, sc.weight.astype(int).values)
    rows.append({'tier': t, 'invites': x.weight.sum(),
                 '%signal': 100*s[s.signal].weight.sum()/max(s.weight.sum(),1),
                 '%geo': 100*g[g.geo].weight.sum()/max(g.weight.sum(),1),
                 'median_step2': np.median(v) if len(v) else np.nan,
                 '%>=260': 100*sc[sc.step2_bucket>=260].weight.sum()/max(sc.weight.sum(),1),
                 '%MD': 100*x[x.degree=='MD'].weight.sum()/x.weight.sum(),
                 '%IMG': 100*x[x.degree.astype(str).str.contains('IMG')].weight.sum()/x.weight.sum()})
p(pd.DataFrame(rows).set_index('tier').round(1).to_string())

hi = recs2[(recs2.n_rank>=6) & recs2.signal.notna()]
lo = recs2[(recs2.n_rank.fillna(0)<=2) & recs2.signal.notna()]
a1,n1 = hi[hi.signal].weight.sum(), hi.weight.sum()
a2,n2 = lo[lo.signal].weight.sum(), lo.weight.sum()
z,pv = sm.stats.proportions_ztest([a1,a2],[n1,n2])
p(f'\n  signal share, most-sought programs  : {a1:.0f}/{n1:.0f} = {100*a1/n1:.1f}%')
p(f'  signal share, low-demand programs   : {a2:.0f}/{n2:.0f} = {100*a2/n2:.1f}%')
p(f'  difference {100*(a1/n1-a2/n2):+.1f} pp, z={z:.2f}, p={pv:.4f}')

# ========================================================================
h('10. GEOGRAPHY')
st = rs.groupby('pkey').state.first()
recs3 = recs.merge(st.rename('state'), left_on='pkey', right_index=True, how='left')
cov = recs3[recs3.state.notna()].weight.sum()
p(f'invites mapped to a state: {cov:.0f}/{recs.weight.sum():.0f} ({100*cov/recs.weight.sum():.0f}%)')
REG = {'CT':'Northeast','ME':'Northeast','MA':'Northeast','NH':'Northeast','RI':'Northeast','VT':'Northeast',
       'NJ':'Northeast','NY':'Northeast','PA':'Northeast',
       'IL':'Midwest','IN':'Midwest','MI':'Midwest','OH':'Midwest','WI':'Midwest','IA':'Midwest','KS':'Midwest',
       'MN':'Midwest','MO':'Midwest','NE':'Midwest','ND':'Midwest','SD':'Midwest',
       'DE':'South','DC':'South','FL':'South','GA':'South','MD':'South','NC':'South','SC':'South','VA':'South',
       'WV':'South','AL':'South','KY':'South','MS':'South','TN':'South','AR':'South','LA':'South','OK':'South','TX':'South',
       'AZ':'West','CO':'West','ID':'West','MT':'West','NV':'West','NM':'West','UT':'West','WY':'West',
       'AK':'West','CA':'West','HI':'West','OR':'West','WA':'West'}
recs3['region'] = recs3.state.map(REG)
g = recs3.dropna(subset=['region'])
rows=[]
for r_, x in g.groupby('region'):
    s = x[x.geo.notna()]; sg = x[x.signal.notna()]
    rows.append({'region': r_, 'invites': x.weight.sum(),
                 '%geo-pref': 100*s[s.geo].weight.sum()/s.weight.sum(),
                 '%signal': 100*sg[sg.signal].weight.sum()/sg.weight.sum(),
                 '%IMG': 100*x[x.degree.astype(str).str.contains('IMG')].weight.sum()/x.weight.sum(),
                 '%DO': 100*x[x.degree=='DO'].weight.sum()/x.weight.sum()})
p(pd.DataFrame(rows).set_index('region').round(1).sort_values('invites',ascending=False).to_string())
p('\nTop states by reported invite volume:')
p(recs3.groupby('state').weight.sum().nlargest(12).to_string())

# ========================================================================
h('11. PRELIM vs CATEGORICAL')
pr = recs[recs.prelim]; ct = recs[~recs.prelim]
for nm, x in [('PRELIM', pr), ('CATEGORICAL (rest)', ct)]:
    s = x[x.signal.notna()]; gg = x[x.geo.notna()]; sc = x.dropna(subset=['step2_bucket'])
    v = np.repeat((sc.step2_bucket+5).values, sc.weight.astype(int).values)
    p(f'{nm:22s} n={x.weight.sum():5.0f} | %signal {100*s[s.signal].weight.sum()/max(s.weight.sum(),1):5.1f} '
      f'| %geo {100*gg[gg.geo].weight.sum()/max(gg.weight.sum(),1):5.1f} '
      f'| median Step2 {np.median(v):.0f} | %IMG {100*x[x.degree.astype(str).str.contains("IMG")].weight.sum()/x.weight.sum():5.1f}')
p('\nIMG share of prelim invites vs categorical invites — the prelim track is the IMG on-ramp.')

# ========================================================================
h('12. IMG-SPECIFIC FACTORS ("II by Date 25-26 IMGs")')
img = store['II by Date 25-26 IMGs'].iloc[1:].copy()
img.columns = ['date','program','signaled','cat_prelim','research_year','prior_prelim','stats','q']
img = img[img.program.notna()]
p(f'structured IMG invite rows: {len(img)}')
img['sig'] = img.signaled.astype(str).str.strip().str.lower().map({'yes':1,'no':0})
img['res'] = img.research_year.astype(str).str.lower().str.contains('yes')
img['pp'] = img.prior_prelim.astype(str).str.lower().str.contains('yes')
p(f"signalled: {img.sig.sum():.0f}/{img.sig.notna().sum():.0f} = {100*img.sig.mean():.1f}%")
p(f"research year (at that program or elsewhere): {img.res.sum()}/{len(img)} = {100*img.res.mean():.1f}%")
p(f"prior prelim year: {img.pp.sum()}/{len(img)} = {100*img.pp.mean():.1f}%")
p(f"Cat vs Prelim: \n{img.cat_prelim.astype(str).str.strip().str.lower().value_counts().to_string()}")
# score from stats string
def sc_of(s):
    m = re.search(r'\b(2\d)\s*[xX]\b', str(s))
    return int(m.group(1))*10+5 if m else np.nan
img['score'] = img.stats.map(sc_of)
img['type'] = np.where(img.stats.astype(str).str.lower().str.contains('non'), 'Non-US IMG',
                np.where(img.stats.astype(str).str.lower().str.contains('img'), 'US IMG', None))
p(f"\nmedian Step2 (from stats string, n={img.score.notna().sum()}): {img.score.median():.0f}")
p(img.groupby('type').agg(n=('score','size'), median_step2=('score','median'),
                          pct_signal=('sig','mean'), pct_research=('res','mean')).round(2).to_string())
p('\nCross-tab research year x signalled:')
ctb = pd.crosstab(img.res, img.sig)
p(ctb.to_string())
if ctb.shape == (2,2):
    odd, pv = stats.fisher_exact(ctb.values)
    p(f'  Fisher exact OR={odd:.2f}, p={pv:.3f}')
p('\nTop programs inviting IMGs:')
p(img.program.astype(str).str.strip().value_counts().head(12).to_string())

# ========================================================================
h('13. POST-INTERVIEW SIGNALS: LETTERS OF INTENT')
loi = store['LOI Outcomes'].iloc[2:].copy()
loi.columns = ['state','statecity','program','signaled','loi_date','resp','resp_date','resp_type'] + \
              [f'x{i}' for i in range(8, 28)]
loi = loi[loi.program.notna() & (loi.program.astype(str).str.strip()!='')]
loi = loi[~loi.program.astype(str).str.contains('Hogwarts', na=False)]


def tally(v):
    """'N +8' -> (no, 9);  'Y+2' -> (yes, 3);  'Y' -> (yes,1)"""
    s = str(v).strip()
    if not s or s.lower() in ('nan','none'): return None, 0
    yes = s.upper().startswith('Y')
    no = s.upper().startswith('N')
    if not (yes or no): return None, 0
    m = re.search(r'\+\s*(\d+)', s)
    return ('Y' if yes else 'N'), 1 + (int(m.group(1)) if m else 0)


ry = rn = 0
sy = sn = 0
for _, r in loi.iterrows():
    k, n = tally(r.resp)
    if k == 'Y': ry += n
    elif k == 'N': rn += n
    k2, n2 = tally(r.signaled)
    if k2 == 'Y': sy += n2
    elif k2 == 'N': sn += n2
p(f'LOI rows logged: {len(loi)} | programs: {loi.program.astype(str).str.strip().nunique()}')
p(f'LOIs that got ANY response : {ry} | no response: {rn}  => response rate {100*ry/(ry+rn):.1f}%')
p(f'LOIs sent to a programme the applicant had signalled: Y={sy}, N={sn} ({100*sy/max(sy+sn,1):.1f}% signalled)')
kinds = loi.resp_type.dropna().astype(str)
p(f'\nOf {len(kinds)} described responses, how many were more than boilerplate:')
gen = kinds.str.contains('generic', case=False).sum()
pos = kinds.str.contains('positive|waitlist|invite|rank|short list|shortlist', case=False).sum()
p(f'  contains "generic": {gen} | contains positive/waitlist/invite/rank language: {pos}')
for s in kinds.head(14): p('   -', s[:120])

lr = store['Letter of intent response'].iloc[1:].copy()
lr.columns = ['program','no_resp','generic','personal','notes','no_loi','x']
for c in ['no_resp','generic','personal']:
    lr[c] = pd.to_numeric(lr[c], errors='coerce').fillna(0)
tot = lr[['no_resp','generic','personal']].sum()
p(f'\nSecond LOI tally sheet ("Letter of intent response"), n={tot.sum():.0f} LOIs:')
p(f'  no response {tot.no_resp:.0f} ({100*tot.no_resp/tot.sum():.0f}%) | '
  f'generic {tot.generic:.0f} ({100*tot.generic/tot.sum():.0f}%) | '
  f'personalised {tot.personal:.0f} ({100*tot.personal/tot.sum():.0f}%)')

# ========================================================================
h('14. POST-INTERVIEW COMMUNICATION & REJECTION/WAITLIST BEHAVIOUR')
pc = store['Post Interview Comms 25-26'].iloc[2:].copy()
pc.columns = ['n','program','tally','date_comm','iv_date','type','pers_gen','rtm','post_loi','second_look','content'] + \
             [f'y{i}' for i in range(11, 13)]
pc = pc[pc.program.notna()]
p(f'post-interview communication reports: {len(pc)}')
if len(pc):
    p(pc[['program','type','pers_gen','rtm','content']].head(20).to_string(index=False, max_colwidth=46))
rw = store['Rejection & Waitlist 25-26'].iloc[3:].copy()
rw.columns = ['program','date','rej','wave','comments','program2','wl_status','final','comments2']
rej = rw[rw.rej.notna()]
p(f'\nrejection reports: {len(rej)} | waitlist reports: {rw.wl_status.notna().sum()}')
if len(rej):
    p(rej[['program','date','rej','wave']].to_string(index=False, max_colwidth=40))

# ========================================================================
h('15. INTERVIEW CHURN — "IIs dropped 25-26"')
dr = store['IIs dropped 25-26']
cnt = 0; texts = []
for r in range(dr.shape[0]):
    for c in range(dr.shape[1]):
        v = dr.iat[r, c]
        if isinstance(v, str) and len(v.strip()) > 3 and not v.strip().startswith("Comment"):
            texts.append(v.strip()); cnt += 1
p(f'non-trivial cells in dropped-interview log: {cnt}')
p('sample:')
for t in texts[3:23]: p('   -', t[:110])

open('analysis_part2.txt','w').write('\n'.join(OUT))
pickle.dump({'rs': rs, 'recs3': recs3, 'recs2': recs2, 'img': img}, open('analysis2.pkl','wb'))
p('\n[written analysis_part2.txt]')
