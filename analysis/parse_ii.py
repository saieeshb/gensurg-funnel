"""Parse the 'II by Date 25-26' sheet into applicant-level invite records."""
import pandas as pd, pickle, re, json, unicodedata, datetime as dt

store = pickle.load(open('sheets.pkl', 'rb'))
df = store['II by Date 25-26']

# ---- date map -------------------------------------------------------------
date_of = {}
for r in range(df.shape[0]):
    v = df.iat[r, 0]
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        date_of[r] = pd.Timestamp(v)
n_released = {}
for r in range(df.shape[0]):
    v = df.iat[r, 1]
    if pd.notna(v):
        try:
            n_released[r] = float(v)
        except (TypeError, ValueError):
            pass

# ---- record regexes -------------------------------------------------------
# tolerate unclosed parens: capture until ')' '}' or end of line
REC = re.compile(r'[({\[]([^)}\]\n]{4,160})[)}\]]?', re.I)
MULT = re.compile(r'^\s*[.,]?\s*(?:[xX]\s*(\d)|\+\s*(\d+))')


def norm(s):
    s = unicodedata.normalize('NFKD', str(s))
    return re.sub(r'\s+', ' ', s).strip()


def parse_flag(body, names):
    """Return True/False/None for a +/- flag."""
    for nm in names:
        m = re.search(r'([+\-−–])\s*' + nm, body, re.I)
        if m:
            return m.group(1) == '+'
        # bare mention without sign, e.g. 'n/ageo'
    return None


def parse_degree(body):
    b = body.lower().replace('-', ' ').replace('.', ' ')
    b = re.sub(r'\s+', ' ', b)
    if re.search(r'non\s*us\s*img|nonus\s*img|non\s*img', b):
        return 'Non-US IMG'
    if re.search(r'\bus\s*img\b|\busimg\b', b):
        return 'US IMG'
    if re.search(r'\bimg\b', b):
        return 'IMG (unspec)'
    if re.search(r'\bus\s*md\b|\busmd\b', b):
        return 'MD'
    if re.search(r'\bmd\b', b):
        return 'MD'
    if re.search(r'\bdo\b', b):
        return 'DO'
    return None


def parse_score(body):
    """Return (bucket_decade, exact_or_None). 24x -> 240s."""
    m = re.search(r'\b(2\d)\s*[xX]\b', body)
    if m:
        return int(m.group(1)) * 10, None
    m = re.search(r'\b(2[0-9]{2})\b', body)
    if m:
        v = int(m.group(1))
        if 190 <= v <= 290:
            return (v // 10) * 10, v
    m = re.search(r'\b(1[89]\d)\b', body)
    if m:
        v = int(m.group(1))
        return (v // 10) * 10, v
    return None, None


records = []
prog_cells = []
for r in range(df.shape[0]):
    if r not in date_of:
        continue
    for c in range(2, df.shape[1]):
        v = df.iat[r, c]
        if not isinstance(v, str) or not v.strip():
            continue
        txt = v.replace(' ', ' ')
        # program name = text before first record paren / first newline w/ paren
        first = re.search(r'[({\[]\s*[+\-−–]?\s*(home|away|signal|sig|geo)', txt, re.I)
        name = txt[:first.start()] if first else txt
        name = name.split('\n')[0]
        flags = {
            'in_person': bool(re.search(r'in[- ]person', txt, re.I)),
            'virtual': bool(re.search(r'\bvirtual\b', txt, re.I)),
            'prelim': bool(re.search(r'\bprelim', txt, re.I)),
            'email_invite': bool(re.search(r'email invite', txt, re.I)),
        }
        clean = re.sub(r'-\s*IN PERSON|IN PERSON( OR VIRTUAL)?|\(email invite\)|PRELIM|\bVIRTUAL\b', '', name, flags=re.I)
        clean = norm(clean).strip(' -–,|')
        prog_cells.append({'row': r, 'col': c, 'date': date_of[r], 'program_raw': clean, 'text': txt, **flags})

        for m in REC.finditer(txt):
            body = m.group(1)
            if not re.search(r'home|away|signal|sig\b|geo', body, re.I):
                continue
            tail = txt[m.end():m.end() + 8]
            mm = MULT.match(tail)
            n = 1
            if mm:
                if mm.group(1):
                    n = int(mm.group(1))            # 'x2' => 2 people total
                else:
                    n = 1 + int(mm.group(2))        # '+2' => 1 + 2 more
            n = min(n, 12)
            deg = parse_degree(body)
            bucket, exact = parse_score(body)
            rec = {
                'date': date_of[r], 'row': r, 'col': c,
                'program_raw': clean,
                'home': parse_flag(body, ['home']),
                'away': parse_flag(body, ['away']),
                'signal': parse_flag(body, ['signal', 'signa', 'sig\\b', 'sig,']),
                'geo': parse_flag(body, ['geo']),
                'degree': deg,
                'step2_bucket': bucket,
                'step2_exact': exact,
                'loi': bool(re.search(r'\bLOI\b', body, re.I)),
                'weight': n,
                'prelim': flags['prelim'],
                'in_person': flags['in_person'],
                'raw': body.strip(),
            }
            records.append(rec)

recs = pd.DataFrame(records)
cells = pd.DataFrame(prog_cells)
print('parsed records (rows):', len(recs), '| weighted invites:', int(recs.weight.sum()))
print('program cells:', len(cells))
print('\ndate range:', recs.date.min().date(), '->', recs.date.max().date())
print('\n--- field completeness (weighted) ---')
for f in ['home', 'away', 'signal', 'geo', 'degree', 'step2_bucket']:
    known = recs[recs[f].notna()].weight.sum()
    print(f'  {f:14s} {known:6.0f} / {recs.weight.sum():.0f}  ({100*known/recs.weight.sum():.1f}%)')
print('\n--- degree ---')
print(recs.groupby('degree').weight.sum().sort_values(ascending=False))
print('\n--- step2 bucket ---')
print(recs.groupby('step2_bucket').weight.sum())
print('\n--- flags (weighted, among known) ---')
for f in ['home', 'away', 'signal', 'geo']:
    sub = recs[recs[f].notna()]
    p = (sub[sub[f]].weight.sum() / sub.weight.sum())
    print(f'  {f:8s} +{100*p:5.1f}%  (n={sub.weight.sum():.0f})')
print('\n--- prelim / in-person ---')
print('prelim weighted:', recs[recs.prelim].weight.sum(), '| in-person:', recs[recs.in_person].weight.sum())
print('LOI mentions:', recs[recs.loi].weight.sum())

recs.to_pickle('recs.pkl'); cells.to_pickle('cells.pkl')

# sanity: show 25 random parsed rows
print('\n--- sample parses ---')
print(recs.sample(20, random_state=1)[['date', 'program_raw', 'home', 'away', 'signal', 'geo', 'degree', 'step2_bucket', 'weight']].to_string(max_colwidth=34))
# unparsed degree examples
bad = recs[recs.degree.isna()]
print(f'\n--- {len(bad)} records w/o degree; samples:')
for s in bad.raw.head(12):
    print('   ', s)
