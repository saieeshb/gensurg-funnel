"""Parse the 2024-2025 workbook: applicant-level records with outcomes."""
import pandas as pd, numpy as np, re, pickle, datetime as dt, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)

SRC = '/Users/chikki/Downloads/2024-2025 Residency Application Spreadsheet.xlsx'
xl = pd.ExcelFile(SRC, engine='openpyxl')
store24 = {n: xl.parse(n, header=None) for n in xl.sheet_names}
pickle.dump(store24, open('sheets24.pkl', 'wb'))

df = store24['APPLICANT INFO']
COLS = ['nick','degree','gradyear','region','step1_attempts','step2','comlex1','comlex2',
        'quartile','aoa','ghhs','m3_surg_honors','n_honored','n_pubs','n_exp','couples',
        'blank16','n_apps','n_ii','yield','sig_yield','invite_locs','n_attended','n_withdrew',
        'n_rejections','n_waitlists','blank26','match_status','blank28']
a = df.iloc[7:].copy()
a.columns = COLS[:a.shape[1]]
a = a[a.nick.notna()]
a = a[~a.nick.astype(str).str.upper().str.contains('MEAN|STD DEV|MEDIAN|IQR', na=False)]
print(f'raw applicant rows: {len(a)}')


def num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '').replace('%', '')
    if s in ('-', '', 'nan', 'v', '?'): return np.nan
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else np.nan


for c in ['step2','comlex1','comlex2','n_apps','n_ii','n_pubs','n_exp','n_honored',
          'step1_attempts','n_attended','n_withdrew','n_rejections','n_waitlists']:
    a[c] = a[c].map(num)

# ---- recover the signal-yield column ----------------------------------
# Google Sheets silently coerced "7/15" -> 2024-07-15. Month = signals converted,
# day = 15 = denominator. Values like "13/15" survived as text (13 is not a month).
def sig_yield(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (dt.datetime, dt.date, pd.Timestamp)):
        return v.month if v.day == 15 else np.nan
    s = str(v).strip()
    m = re.match(r'^(\d{1,2})\s*/\s*15$', s)
    if m: return float(m.group(1))
    m = re.match(r'^(\d{1,2})$', s)
    if m and int(m.group(1)) <= 15: return float(m.group(1))
    return np.nan


a['sig_converted'] = a.sig_yield.map(sig_yield)
recov_date = a.sig_yield.map(lambda v: isinstance(v, (dt.datetime, dt.date, pd.Timestamp))).sum()
print(f'signal-yield recovered: {a.sig_converted.notna().sum()} '
      f'({recov_date} of them rescued from date-coercion)')

# ---- normalise categoricals -------------------------------------------
def deg(x):
    s = str(x).strip().upper()
    if s.startswith('MBBS') or 'IMG' in s: return 'IMG'
    if s.startswith('DO'): return 'DO'
    if s.startswith('MD'): return 'MD'
    return np.nan


a['deg'] = a.degree.map(deg)


def region(x):
    s = str(x).strip().upper()
    if 'IMG' in s or 'CARIB' in s: return 'IMG'
    for k, v in [('NORTHEAST','Northeast'), ('MID-ATL','Mid-Atlantic'), ('MID ATL','Mid-Atlantic'),
                 ('MIDATL','Mid-Atlantic'), ('MIDWEST','Midwest'), ('SOUTH ATL','South Atlantic'),
                 ('SOUTH','South'), ('WEST','West'), ('NEW ENGLAND','Northeast'),
                 ('CENTRAL','Midwest'), ('PACIFIC','West'), ('SOUTHWEST','West')]:
        if k in s: return v
    return np.nan


a['reg'] = a.region.map(region)


def yn(x):
    s = str(x).strip().upper()
    return True if s.startswith('Y') else (False if s.startswith('N') else np.nan)


for c in ['aoa','ghhs','m3_surg_honors','couples']:
    a[c + '_b'] = a[c].map(yn)


def status(x):
    s = str(x).strip().lower()
    if not s or s == 'nan': return np.nan
    if 'categor' in s: return 'Categorical'
    if 'prelim' in s: return 'Preliminary'
    if 'unmatch' in s or 'soap' in s or 'did not' in s: return 'Unmatched'
    return np.nan


a['status'] = a.match_status.map(status)
a['quart'] = a.quartile.astype(str).str.extract(r'(\d)')[0].astype(float)

# recompute yield rather than trusting the sheet's formula cells
a['yield_calc'] = a.n_ii / a.n_apps
a.loc[(a.n_apps <= 0) | a.n_apps.isna(), 'yield_calc'] = np.nan
a['sig_rate'] = a.sig_converted / 15.0

print('\n--- completeness ---')
for c in ['deg','step2','n_apps','n_ii','yield_calc','sig_converted','status','n_pubs',
          'aoa_b','quart','reg','couples_b']:
    print(f'  {c:16s} {a[c].notna().sum():4d} / {len(a)}  ({100*a[c].notna().mean():.0f}%)')

print('\n--- degree ---'); print(a.deg.value_counts().to_string())
print('\n--- match status ---'); print(a.status.value_counts().to_string())
print('\n--- key numerics ---')
print(a[['step2','n_apps','n_ii','yield_calc','sig_converted','n_pubs','n_exp']]
      .describe().round(2).to_string())
a.to_pickle('appl24.pkl')
print('\n[saved appl24.pkl]')
