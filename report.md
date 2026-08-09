# What Actually Drives a General Surgery Interview

### Two cycles of the crowdsourced applicant spreadsheets, analysed together

**Sources:**
`2024-2025 Residency Application Spreadsheet.xlsx` (19 tabs) — **361 applicant records with outcomes**
`Official 2026-2027 Gen Surgery Spreadsheet.xlsx` (30 tabs) — **1,154 interview-invite records, 315 programs**

**Analysis date:** 9 August 2026
**Live version:** [surgery.saieesh.dev](https://surgery.saieesh.dev) · **IMG-specific analysis:** [surgery.saieesh.dev/img.html](https://surgery.saieesh.dev/img.html)

---

## 0. What the two workbooks are, and why they don't stack

The two files record **different units of observation**. This is the single most important
methodological fact in the report, and the easiest thing to get wrong.

| Property | 2024–25 workbook | 2025–26 workbook |
|---|---|---|
| Unit of observation | **the applicant** | **the invitation** |
| Records parsed | 361 | 1,154 |
| Step 2 CK | exact score | decade bucket only |
| Application counts | yes | no |
| Signal conversion (x/15) | **yes — direct measurement** | per-invite flag only |
| Match outcome | 185 reported | none |
| Invitation dates | none | every record |
| Program identity | region only | 315 programs |
| Applicant identity | pseudonym | **none at all** |

They are complementary, not comparable head-to-head. The 2024–25 file answers *what happens to an
applicant*; the 2025–26 file answers *who gets invited, from where, and when*. Any chart that
implies a continuous two-year trend across them is a defect.

The 2026–27 tabs in the second workbook were **empty scaffolding** at the time of analysis —
`IV IMPRESSIONS 26-27` has 407 rows and one comment in it; `2nd Look 26-27` has zero filled
responses. Nothing here is derived from them.

---

## 1. Executive summary

Twelve findings, ordered by how much confidence the data supports.

| # | Finding | Strength |
|---|---|---|
| 1 | **Interview count is nearly the whole outcome.** 9.1× the odds of a categorical match per five extra interviews. 10+ invites → 91% categorical; ≤2 → 0%. | Definitive within sample |
| 2 | **Signals amplify a strong application; they do not rescue a weak one.** Conversion rises from **4.0/15** below 240 to **9.1/15** above 270 (r = +0.44, p < 10⁻⁵). | Strong — direct measurement |
| 3 | **Program signalling is the dominant controllable factor**, worth ~3.9× the odds of an invite (tier-matched OR 3.85, 95% CI 1.77–8.35), confirmed by two independent derivations. | Strong |
| 4 | **Signal value scales with program desirability.** Signalled share of invitees: 30% at low-demand → 55% mid → **75% at the most sought-after**. | Strong; survives clustering, bootstrap, multiplicity |
| 5 | **Past ~60 applications you are buying nothing.** Marginal return falls from +1.9 interviews per 10 applications to **zero**; ρ(apps, yield) = −0.57. | Strong, but confounded — see §16 |
| 6 | **The season is one day long.** 35% of all invites landed Wed 22 Oct 2025, the APDS universal release date. 45% of volume fell on Wednesdays. | Definitive |
| 7 | **Publications do nothing measurable for interview count** (IRR 1.01, 95% CI 0.93–1.09, p = 0.85), holding score, pathway, AOA and class rank constant. | Strong null, power-bounded |
| 8 | **Step 2 CK is the strongest measured input to interviews** — 1.22× per 10 points — but it works *through* interview count, not independently at the rank stage. | Strong |
| 9 | **Non-US IMGs face a different market, not a harder one.** 57% of their invites are prelim (vs 1.9% for MDs); their invitees score **10 points higher** than MD invitees. | Strong on invite side |
| 10 | **Prelim and categorical are separate labour markets.** IMGs are 78% of prelim invites and 8% of categorical (χ² = 318, p ≈ 3×10⁻⁷¹). | Definitive |
| 11 | **Away rotations and signals travel together; home rotations don't.** 86% of away-rotators also signalled (OR 5.75); home-program invitees signalled at base rate (OR 0.85, p = 0.64) — a free signal. | Strong |
| 12 | **Post-interview letters are close to noise.** LOI response rate 34%; only 30% personalised. Two independent tallies agree. | Moderate |

**The single most important structural fact:** the intraclass correlation for signal status *within
programs* is 0.415. Programs differ from each other far more than applicants within a program
differ from one another. Which program you apply to is a bigger lever than marginal changes to
your application.

---

## 1a. The funnel, in one table

From the 165 applicants in 2024–25 who reported all three stages:

| Stage | Median | Middle half (IQR) |
|---|---|---|
| Applications sent | **75** | 57 – 110 |
| Interview invites | **14** | 10 – 18 |
| Interviews attended | **12** | 7 – 16 |
| Matched categorical | **65.4%** | of the 185 reporting an outcome |

93.8% of invitations are taken up, so the decisive narrowing happens between applications and
invites. That is where Stage 02 of the analysis concentrates — and where the levers are.

*Caveat: the spine subsample skews successful (median 14 invites against 11 for all 346 applicants
reporting apps and invites), because reporting attendance requires having had interviews to
attend.*

---

## 1b. What fifteen signals actually bought — the direct measurement

The 2024–25 workbook contains a column headed `II by Signals x/15`. It looked unusable: most cells
held dates like `2024-07-15`. **They are not dates.** Google Sheets silently coerced entries typed
as `7/15` into 7 July. The day component is always 15 — the denominator — and the month component
is the numerator. Values above twelve survived as text, because `13/15` is not a valid month, which
is exactly the pattern that confirms the mechanism rather than merely fitting it.

```python
def sig_yield(v):
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return v.month if v.day == 15 else nan   # "7/15" -> 2024-07-15
    m = re.match(r'^(\d{1,2})\s*/\s*15$', str(v).strip())
    return float(m.group(1)) if m else nan       # "13/15" survived as text
```

This recovered **254 usable records**, 233 of them rescued from date-coercion.

| Measure | Value |
|---|---|
| Mean converted | **6.24 of 15 (41.6%)** |
| Median | 6 of 15 (IQR 3–9) |
| Converted zero | 3.9% |
| Converted 10+ | 19.3% |

**By score band — the finding that overturned my prior:**

| Step 2 CK | n | Mean converted of 15 |
|---|---|---|
| <240 | 47 | **4.04** |
| 240–249 | 57 | 5.70 |
| 250–259 | 76 | 6.41 |
| 260–269 | 50 | 7.94 |
| 270+ | 18 | **9.11** |

r = +0.444, n = 248, p < 10⁻⁵.

I went into the 2025–26 data expecting to find signals *compensating* for a low score. Among
invitees they don't (§6), and that hypothesis died under Holm correction. The 2024–25 conversion
data then showed the **opposite** relationship, and it is much stronger. A 270 applicant converts
more than twice as many signals as a sub-240 applicant. **The signal is a multiplier on an
application, not a substitute for one.**

**By pathway:**

| Pathway | n | Mean converted of 15 |
|---|---|---|
| US MD | 196 | 6.51 |
| DO | 42 | 6.36 |
| IMG | 15 | **2.40** |

MD vs IMG p = 0.00003. MD vs DO p = 1.00 — DOs convert signals exactly as well as MDs.

---

## 1c. The yield curve — where the marginal application dies

346 applicants reporting both application and invitation counts. Median 79 applications, 11
invites, 13.1% hit rate.

| Applications sent | n | Median apps | Median invites | **Median yield** |
|---|---|---|---|---|
| <40 | 35 | 32 | 9 | **30.0%** |
| 40–59 | 69 | 51 | 13 | **24.5%** |
| 60–79 | 80 | 70 | 13 | **18.5%** |
| 80–119 | 75 | 97 | 12 | **13.1%** |
| 120+ | 87 | 170 | 7 | **3.5%** |

- Spearman ρ(applications, **yield**) = **−0.572**, p = 1.7×10⁻³¹
- Spearman ρ(applications, **invites**) = **−0.154**, p = 0.004 — *negative*

Marginal return, fitted within band:

| Band | n | Invites gained per 10 extra applications |
|---|---|---|
| ≤60 apps | 104 | **+1.91** |
| 61–120 apps | 155 | **−0.30** |
| >120 apps | 87 | +0.08 |

**This is confounded and cannot be made causal.** Weaker applicants apply more broadly *because*
they expect a low hit rate; application count is partly a self-assessment of competitiveness. What
survives that objection is narrower: the marginal slope flattens to zero past roughly sixty, and
the broad-application strategy demonstrably does not rescue the people using it — the 120+ group
sends more than twice the median list and ends with **fewer** interviews than the 40–59 group.

It does **not** follow that cutting a list to forty raises anyone's hit rate. That is the natural
misreading and nothing here supports it.

---

## 1d. What predicts an interview — the applicant-level model

Poisson GLM of interview count, HC1 robust standard errors, n = 337.

| Factor | Multiplier (IRR) | 95% CI | p |
|---|---|---|---|
| Step 2 CK, per 10 points | **1.22×** | 1.15 – 1.30 | <0.001 |
| M3 surgery honors | **1.22×** | 1.04 – 1.43 | 0.015 |
| Applications sent, per log unit | **1.19×** | 1.04 – 1.35 | 0.009 |
| AOA | 1.17× | 0.99 – 1.38 | 0.060 |
| Publications, per log unit | **1.01×** | 0.93 – 1.09 | **0.85** |
| Top-quartile class rank | 0.92× | 0.78 – 1.09 | 0.34 |
| IMG pathway | **0.25×** | 0.17 – 0.38 | <0.001 |

Two results deserve emphasis.

**Publications do nothing measurable.** This is the clearest gap between what applicants optimise
and what the data rewards. Bounded by power (§12.8): it says publication count is not a *large*
lever, not that the effect is zero.

**The IMG pathway carries a 0.25× penalty** — a three-quarters reduction in expected interviews,
holding score, publications, AOA, class rank and application count constant. That is the single
largest coefficient in the model, and it is a property of the pathway, not of the applicant.

---

## 1e. What predicts matching

185 applicants reported an outcome: 121 categorical, 21 preliminary, 43 unmatched.

| Interview invites | n | Matched categorical |
|---|---|---|
| 0 | 39 | **0%** |
| 1–2 | 5 | **0%** |
| 3–5 | 10 | 40.0% |
| 6–9 | 15 | 73.3% |
| 10+ | 113 | **91.2%** |

≤5 invites: 7.4% categorical. ≥10 invites: 91.2%. z = −10.55, p = 5×10⁻²⁶.

Logistic model (HC1 robust, n = 178):

| Factor | OR | 95% CI | p |
|---|---|---|---|
| Interviews, per 5 | **9.08** | 4.71 – 17.49 | <0.001 |
| AOA | **6.76** | 1.70 – 26.98 | 0.007 |
| Step 2 CK, per 10 points | **0.58** | 0.37 – 0.92 | 0.020 |
| IMG pathway | 0.58 | 0.05 – 6.31 | 0.66 |
| DO pathway | 0.44 | 0.09 – 2.21 | 0.32 |

**On the negative Step 2 coefficient:** do not read it as "a higher score hurts". It is collider
stratification — holding interview count fixed compares a 270 who managed only eight interviews
against a 240 who managed eight, and the former has something else going wrong or aimed at harder
programs. The causal path for score runs *through* interview count, where §1d measures it at 1.22×
per ten points.

**Outcome reporting is the weakest link in the whole report.** Only 51% reported an outcome, months
after the fact, voluntarily. People who match have reason to return and update; people who don't
also return, to ask what to do next. The direction of that bias is genuinely unclear, which is
worse than a bias whose direction is known. Read the 65.4% categorical rate as a within-sample
figure, never as a match rate for General Surgery.

---

## 1f. Non-US IMGs: a different market

Full treatment at [surgery.saieesh.dev/img.html](https://surgery.saieesh.dev/img.html). The
headline results:

| Measure | Non-US IMG | US MD |
|---|---|---|
| Share of invitations that were **preliminary** | **56.6%** | 1.9% |
| Median Step 2 CK **among invitees** | **265** | 255 |
| Share of invitees at 260+ | **57.8%** | 44.9% |
| Geographic tie on the invitation | **45.7%** | 74.7% |
| Median interview yield (IMG, 2024–25) | **1.2%** | 15.6% |
| Median applications sent | **137** | 74 |
| Mean signals converted of 15 | **2.40** | 6.51 |

The score comparison is the important one: Mann–Whitney p = 0.00004. **The non-US IMGs who get
invited score ten points higher than the US MDs who get invited.** The filter is not tighter, it is
different.

Within the IMG group, Step 2 CK is the only factor that moves interview count — IRR **1.64×** per
ten points, against 1.22× in the all-applicant model (n = 30, descriptive).

Outcomes for the 15 IMGs reporting: 2 categorical, 6 preliminary, 7 unmatched. Against 70.3%
categorical for MDs. **Fifteen people cannot support a rate**, and the 2024–25 workbook does not
separate US from non-US IMGs — it has one `IMG` category. Only the 2025–26 invitation log
distinguishes them, so the invite-side findings above are genuinely about non-US IMGs while the
outcome-side ones are about a mixed group.

The uncomfortable finding: IMG applicants brought a **higher** median publication count (9 vs 6,
mean 16 vs 11, p = 0.03) and an essentially identical median Step 2 CK (253.5 vs 253.0), sent
nearly twice as many applications, and converted them at one-thirteenth the rate. Whatever
produces that gap, it is not the credentials these workbooks record.

---

## 1g. Two-year comparison — what is genuinely comparable

**Program desirability** (`RANK SPOTS` in both workbooks, 314 programs in common):

- Spearman ρ = **+0.552**, p = 2×10⁻²⁶ — the broad ordering persists.
- But only **9 of the top 25** most-ranked programs held their place.
- Gini of demand: 0.505 (24–25) → 0.533 (25–26). Top-25 share: 28.4% → 31.1%.

Rank slots are small counts driven by who filled in the sheet. Individual movements — UCLA going
30 → 6 — are almost certainly a collapse in respondents, not in demand. **Read the correlation, not
the deltas.**

**Step 2 CK** is comparable only with care:

| Population | n | Median | IQR |
|---|---|---|---|
| All 2024–25 applicants | 352 | 253 | 243–261 |
| 2024–25 applicants with ≥1 invite | 299 | 253 | — |
| 2025–26 invitees | 1,141 | 255 | 255–265 |

The gap between rows 1 and 3 is the **interview filter**, not year-over-year drift — different
populations in different years. The like-for-like comparison (row 2 vs row 3) differs by 2 points.

---

## 2. Invite-side finding — How much a signal is actually worth

### 3.1 Tier-matched estimate (the internally valid one)

Nine programs appear in both `Comm 25-26` (with application denominators) and the invite log. Same programs, same cohort, both sides carrying the same reporting bias:

|  | n | Signalled | Share |
|---|---|---|---|
| Applications to these 9 programs | 373 | 170 | 45.6% |
| Invites from these 9 programs | 38 | 29 | **76.3%** |

Odds of having signalled rise from 0.837 among applicants to 3.222 among invitees.

> ### Signal odds ratio = **3.85** (95% CI 1.77 – 8.35)

### 3.2 Structural estimate (independent route)

General Surgery gave 15 signals in 2025–26 (confirmed inside the sheet — `APPLICANT INFO` tracks *"II by Signals x/15"*, and chat posts read *"0/15"*, *"1/15"*). If an applicant sends N applications, base rate = 15/N:

| Applications sent | Base rate | Implied OR |
|---|---|---|
| 40 | 37.5% | 1.94 |
| 50 | 30.0% | 2.72 |
| **60** | **25.0%** | **3.50** |
| **70** | **21.4%** | **4.28** |
| 80 | 18.8% | 5.05 |
| 100 | 15.0% | 6.61 |

Typical General Surgery applicants send 60–70 applications. That window gives OR ≈ 3.5–4.3 — **the tier-matched estimate of 3.85 lands right inside it.** Two methods with different assumptions and different failure modes converge.

### 3.3 Where a signal buys the most

Combining the observed signalled-share of invitees with the *program-specific* base rates measured in §0:

| Program tier | Invites | % invitees signalled | Base signal rate | Implied OR |
|---|---|---|---|---|
| Most-sought (6+ rank slots) | 185 | 74.6% | 50.1% | 2.92 |
| Mid (3–5 rank slots) | 128 | 54.7% | 33.9% | 2.35 |
| Low-demand (0–2 rank slots) | 835 | 49.1% | 17.8% | **4.46** |

This is the counter-intuitive result, and it's the most strategically useful thing in the report.

**In odds terms, a signal does the most work at programs almost nobody signals.** At an academic magnet, half the applicant pool signalled — your signal buys entry to a crowded room. At a community programme where 18% signalled, the same signal is a much rarer object.

But read the two columns together. At the most-sought programs, 75% of everyone invited had signalled — meaning **without a signal you are competing for a quarter of the seats**. High odds-ratio at low-demand programs, high *necessity* at high-demand ones. Those are different arguments for spending the same signal, and they point in opposite directions.

---

## 3. Invite-side finding — Signal value scales with how much a program is wanted

Joining the invite log to `RANK SPOTS 25-26` (112 programs matched, 1,110 rank slots reported):

| Program tier | Invites | % signalled | % geo | Median Step 2 | % ≥260 |
|---|---|---|---|---|---|
| 1–2 rank slots | 128 | **29.7%** | 80.5% | 255 | 28.1% |
| 3–5 rank slots | 128 | **54.7%** | 69.6% | 255 | 42.9% |
| 6+ rank slots (most sought) | 188 | **74.6%** | 75.9% | 265 | 56.7% |

A clean monotone gradient, 30% → 55% → 75%.

**Robustness — I tried hard to break this:**

- **Program-clustered SEs:** OR 3.14 (95% CI 1.49–6.63), p=0.003. Clustered SEs run 36% wider than naive; the effect survives.
- **Cluster bootstrap** (4,000 resamples over programs): gap 24.8 pp, 95% CI 8.6–39.4 pp, P(gap≤0)=0.0013.
- **Excluding the 22 Oct mass-release day** (35% of the data): gap narrows from +24.6 pp to **+17.0 pp** but holds.
- **Holm correction** across six headline tests: survives at p<0.001.
- **Simpson's paradox check:** the near-zero pooled score↔signal correlation isn't hiding opposite within-tier signs (most-sought r=+0.015; rest r=−0.052).

---

## 4. Invite-side finding — The season is essentially one day

| Date | Invites | Share |
|---|---|---|
| **Wed 22 Oct 2025** | **409** | **35.4%** |
| Thu 23 Oct | 61 | 5.3% |
| Tue 21 Oct | 54 | 4.7% |

409 invites across **116 programs in a single day.** The three biggest days hold 45.4% of the season.

The sheet identifies the cause: an **APDS-recommended universal release date** (a participant links the APDS guidance document in `PCPD Q&A 25-26` r147c3). It is not a mandate — a program administrator in the same thread says *"I'm not 100% sure where the 'universal release date' came from... it hasn't been something that was shared with me directly."*

**Day-of-week concentration:**

| Day | Share |
|---|---|
| Wednesday | **45.4%** |
| Tuesday | 17.8% |
| Thursday | 13.1% |
| Monday | 11.4% |
| Friday | 10.5% |
| Weekend | 1.8% |

**Cumulative timeline:**

| Milestone | Date |
|---|---|
| First invite | 25 Sep 2025 |
| 10% landed | 10 Oct |
| 25% landed | 20 Oct |
| **50% landed** | **22 Oct** |
| 75% landed | 27 Oct |
| 90% landed | 5 Nov |
| 95% landed | 21 Nov |
| Last recorded | 23 Jan 2026 |

### 5.1 Signal share is not flat across the season

| Week of | Invites | % signalled |
|---|---|---|
| 22 Sep | 17 | 47.1% |
| 29 Sep | 54 | 42.6% |
| 6 Oct | 55 | **30.9%** |
| 13 Oct | 143 | 37.8% |
| **20 Oct (release)** | **592** | **60.6%** |
| 27 Oct | 145 | 57.9% |
| 3 Nov | 57 | 47.4% |
| 10 Nov | 18 | **27.8%** |

Not a trend — a **spike**. Early-trickle and late-season invites are predominantly *unsignalled*; the release-day bolus is where signals cash out. The weak positive correlation (r=+0.072, p=0.015) badly under-describes this shape.

Practically: **if you are unsignalled at a program, your invite — if it comes — most likely arrives before or well after the release day, not on it.** Sitting at zero on 22 October is not the verdict it feels like.

---

## 5. Invite-side finding — Step 2 CK is a gate, not a ladder

| Bucket | Invites | Share | Cumulative |
|---|---|---|---|
| ≤229 | 6 | 0.5% | 0.5% |
| 230s | 60 | 5.3% | 5.8% |
| 240s | 212 | 18.6% | **24.4%** |
| 250s | 369 | 32.3% | 56.7% |
| 260s | 364 | 31.9% | 88.6% |
| 270s+ | 130 | 11.4% | 100% |

Median **255**; 43.3% at 260+; **24.4% below 250**; 5.8% below 240.

Now the important part. Score has **no relationship to signal status** among invitees:

- Cluster-robust logistic model: OR 0.945 per 10 points (95% CI 0.83–1.08), **p=0.41**
- Point-biserial correlation: r=−0.030, p=0.32
- Signal share ≤249 vs ≥260: 61.2% vs 55.1%, +6.1 pp, **fails Holm correction** (adjusted p=0.199)

I'd expected to find that low scorers lean on signals to compensate. **The invite data does not support it, and the 2024–25 conversion data reverses it** — see §1b, where signal conversion rises from 4.0/15 below 240 to 9.1/15 above 270 (r = +0.44). Signals multiply an application; they do not substitute for one. The most defensible reading is the one PDs describe in their own words in the sheet:

> **PC:** *"we look at the average for the specialty and that is what we base our cutoff at. The minimum score is the minimum regardless if you're an MD or IMG. We use the COMLEX average for DO applicants."*

> **PD/PC:** *"Our program does not have any cutoff for USMLE. We deactivate anyone who has failed an exam but otherwise scores do not play a factor."*

> **PD/PC:** *"a signal, away rotation, and good feedback does not equal an interview. You likely did not get screened out, but truthfully I'm guessing you just didn't score high enough in their review to get an interview invite."*

Score gets you past the filter. **Past the filter, it stops differentiating** — which is exactly what a distribution with a quarter of invitees below 250 looks like.

---

## 6. Invite-side finding — Prelim and categorical are different markets

| | IMG | Non-IMG | IMG share |
|---|---|---|---|
| **Prelim invites** | 63 | 18 | **77.8%** |
| **Categorical invites** | 83 | 961 | **8.0%** |

χ² = 318.4, **p ≈ 3×10⁻⁷¹**.

| Track | n | % signalled | % geo | Median Step 2 |
|---|---|---|---|---|
| Prelim | 83 | 59.0% | **45.7%** | **265** |
| Categorical | 1,071 | 53.4% | 76.1% | 255 |

Prelim invitees carry **higher** scores (median 265 vs 255) and far weaker geographic ties (46% vs 76%). That is the signature of a market where applicants take what they can get, geography be damned — and where a strong score does not convert into a categorical seat.

The IMG chat tab quantifies the signal problem for this group directly:

- *"6 cat, 0/15 signals. I think signals did not matter for us non us imgs"* — followed by *"non us img 26x, 10 pubmed pubs, research year, reached out to mentors to vouch for me"*
- *"Im on 0/15 :( +4"* (five people)
- *"whats ur yield on signals im at 1/15 is that normal"* → *"1/15 and its a prelim, honestly I think signals matter very little for Non-US imgs"*

Cross-referencing the structured IMG tab (84 rows): 65.5% had a research year, median Step 2 255, 56% signalled — and **research year showed no association with getting signalled invites** (Fisher OR 0.85, p=0.82).

**Reading it straight:** for Non-US IMGs, signals appear to convert at near-zero rates, and the differentiator in the successful cases is research volume plus personal advocacy, not the signal. That one applicant with 6–7 categorical invites went 0/15 on signals and named *"reached out to mentors to vouch for me"* as the mechanism.

---

## 7. Invite-side finding — Geography splits four ways

| Pathway | n | % geo-preferred |
|---|---|---|
| **DO** | 151 | **84.1%** |
| MD | 815 | 74.7% |
| US IMG | 63 | 69.8% |
| **Non-US IMG** | 81 | **45.7%** |

χ² = 42.07, **p < 10⁻⁶** (survives Holm).

Overall, 73.9% of invitees had a stated geographic preference for the program that invited them (95% CI 71.3–76.4). Geography is the second-most-common attribute in the entire dataset after being a US MD.

By region:

| Region | Invites | % geo-pref | % signalled | % IMG | % DO |
|---|---|---|---|---|---|
| Northeast | 178 | **81.8%** | 59.1% | 15.2% | 7.9% |
| South | 147 | 75.3% | 54.4% | 17.0% | 15.6% |
| Midwest | 84 | 67.9% | 42.2% | **1.2%** | 11.9% |
| West | 62 | 67.2% | 60.7% | 6.5% | 11.3% |

The Midwest is close to an IMG dead zone in this data (1.2% of invites) while running the lowest signal share (42.2%) — a region where fewer people signal and non-US graduates barely appear. Top states by volume: NY (72), FL (38), NJ (31), MA (28), PA (22).

*Caveat: state mapping covers 41% of invites — those whose program matched the RANK SPOTS roster.*

---

## 8. Invite-side finding — Rotations: away and home behave differently

| Profile | Invites | Share of all | % also signalled |
|---|---|---|---|
| Did an away there | 76 | 6.6% | **85.5%** |
| Home program | 42 | 3.7% | **50.0%** |
| Neither | 1,030 | 89.7% | 51.7% |

- **Away rotation × signal:** Fisher OR **5.75**, p<10⁻⁵. In the cluster-robust model, OR 6.49 (95% CI 2.98–14.13).
- **Home program × signal:** OR **0.85**, **p=0.64** — indistinguishable from base rate.

Applicants who invest in an away rotation almost always spend a signal there too — 86% do. Applicants at their home programme **do not**, and the sheet gives no sign it costs them. That is a rational, and apparently correct, allocation: your home program already knows you.

One PD confirms the mechanism explicitly:

> *"We have a shortlist for applicants that have done rotations with us so they wouldn't get screened out based on scores alone."*

**An away rotation buys a score-filter exemption.** That is a materially different good from what a signal buys.

---

## 9.  What program directors said in their own words

`PCPD Q&A 25-26` holds 88 KB of PD/PC answers — 391 substantive text blocks, 170 touching selection criteria. The load-bearing quotes:

**On signal oversubscription — the most important sentence in the workbook:**
> *"For our program, no change in the number of interviews. However, it looks like a greater fraction of our invites are going to students that signaled. **We had more students signal than interview spots.**"*

**On signals affecting rank, not just invites:**
> **PD:** *"I consider signals in both interview and rank"*

**On signals not being a guarantee:**
> *"programs have the right to reject you even if you signalled them, its part of life... they can't accept everyone that signalled them"*

**On the "holistic review" question, from an applicant with 0 invites, 21 posters, 5 publications, and a Step 2 fail followed by a 272 retake:**
> *"Are you guys just being nice by saying the application process is 'holistic'? ... It's hard for me to believe you guys don't have an automatic filter for fails."*

That question went unanswered in the sheet. The `deactivate anyone who has failed an exam` quote in §6 is the closest thing to a reply — and it suggests the answer is no, they aren't being nice.

**On post-interview communication:**
> **PD:** *"making phone calls or emails to say RTM for me is a violation of NRMP communication rules. WE DO NOT reach out to applicants to tell them that they are RTM. Not getting a RTM does NOT mean that a program is not interested in you."*

---

## 10.  Post-interview letters are close to noise

Two independent tabs, same answer.

**`LOI Outcomes`** — 69 rows, 64 programs:

| Outcome | Count | Share |
|---|---|---|
| Any response | 42 | **34.4%** |
| No response | 80 | 65.6% |

**`Letter of intent response`** — 40 LOIs:

| Outcome | Share |
|---|---|
| No response | 42% |
| Generic | 28% |
| Personalised | 30% |

**84.7% of LOIs went to programs the applicant had already signalled** — meaning the LOI is overwhelmingly a *second* touch on an existing bet, not a way to open a new door.

Of 38 described responses, only 2 explicitly said "generic," while 22 contained positive/waitlist/invite/rank language — but the described responses are the memorable ones, which is precisely a selection effect. The unglamorous denominator is the 42–66% that got nothing.

Two entries that matter more than the averages:

> *"Soft Rejection by PC. **Focusing on applicants that signalled this season.**"*
> *"received IV invite day later"*

The first is a program stating its filter out loud. The second is the tail everyone is playing for.

**`Rejection & Waitlist 25-26`** logs 20 rejections spanning 25 Sep 2025 – 5 Mar 2026, mostly tagged "R-wave" — programs rejecting in batches. Notably, one arrived on **25 September**, one day after programs got ERAS access. Some programs screen and reject within 24 hours.

---

## 11. Things an advanced analysis needs that are easy to skip

You asked what might be missing. These are the checks that separate a real analysis from a spreadsheet summary — each one either changed a conclusion or bounded one.

### 12.1 The denominator problem
Covered in §0. **Without it, every percentage in this report is uninterpretable.** Most analyses of this sheet report "54% of invitees signalled" as if it means something on its own. It doesn't.

### 12.2 Clustering — records are not independent
One program contributes up to 26 invites, all filtered by one committee's rule. Mean 3.7 invites/program; 91 programs with ≥5 invites hold 56% of the data.

- **ICC = 0.415** → design effect **2.48**
- **Effective sample size ≈ 465, not 1,154**
- Clustered SEs run **36% wider** than naive ones

The ICC is itself a finding: **program-level heterogeneity dominates applicant-level variation.**

*Applicant identity is recorded nowhere in the sheet, so applicant-level clustering cannot be corrected at all. This is the single largest un-fixable limitation.*

### 12.3 Multiplicity
Six headline tests, Holm-corrected:

| Test | Raw p | Holm p | Survives |
|---|---|---|---|
| Signal share: most-sought vs rest | <10⁻⁵ | <10⁻⁵ | **YES** |
| Geo-preference differs by degree | <10⁻⁵ | <10⁻⁵ | **YES** |
| IMG share: prelim vs categorical | <10⁻⁵ | <10⁻⁵ | **YES** |
| Signal share rises over season | 0.0145 | 0.0435 | **YES** |
| Signal share: ≤249 vs ≥260 Step 2 | 0.0996 | 0.1993 | no |
| Signal use differs by degree | 0.8444 | 0.8444 | no |

**Two plausible-looking findings died here** — including the score-compensation hypothesis I expected to confirm.

### 12.4 Selection bias, named and directed
| Bias | Mechanism | Direction |
|---|---|---|
| **Volunteer** | People post events, not non-events | No absolute probability is estimable |
| **Success** | 15 invites = 15 posts; 0 invites = 0 posts | Inflates apparent credentials |
| **Survivor** | December posters are those still searching | Late-season stats unrepresentative |
| **Audience** | Sheet skews US-MD, US-based | Non-US IMG under-sampled |
| **Recall** | "No signal" may go unmentioned | Signal share possibly overstated |

**Coverage:** ~1,700–1,800 categorical PGY-1 positions across ~350 programs, at ~10–12 interviews per position → ~18,000–22,000 national invites. This sheet holds 1,154. **Coverage ≈ 5–6%.** *(External anchors, not from the sheet.)*

This is why the tier-matched contrast in §3.1 is the load-bearing estimate — both sides of that comparison carry the same bias, so it largely cancels.

### 12.5 Missing-data mechanism
Signal field missing in only 0.5% of records. Signal share among records missing degree (53.6%) or geo (56.2%) tracks the overall 53.8%. Missingness isn't concentrated where it would distort. Treat as missing-at-random.

### 12.6 Robustness to the dominant data point
35% of records come from one day. Re-running the core result without 22 October: the desirability gradient narrows from +24.6 pp to +17.0 pp but survives. **One day cannot be allowed to be the whole finding, and here it isn't.**

### 12.7 Simpson's paradox
Checked the pooled null score↔signal correlation for opposite within-tier signs. None (+0.015 vs −0.052). The null is real, not an artefact of aggregation.

### 12.8 Power
| Effect to detect | Power |
|---|---|
| 5 pp | 0.23 |
| 10 pp | 0.70 |
| 15 pp | 0.96 |
| 20 pp+ | ~1.00 |

**This dataset cannot see effects under ~10 pp.** Every null above (score, geo, home, DO status) is consistent with a real effect below that threshold. *Absence of evidence, not evidence of absence.*

### 12.9 Two independent estimation routes
§3.1 (tier-matched, empirical) and §3.2 (structural, 15/N) rest on different assumptions and fail differently. They converge on OR ≈ 3.5–4.3. Convergence is worth more than either estimate alone.

### 12.10 Falsifiable predictions
Anyone can describe last year. §12.2 commits to ten predictions checkable against the 26-27 sheet by December.

---

## 12. The 2026–27 cycle

### 13.1 Calendar

The 2027 ERAS cycle repeats the 2026 weekday structure **364 days later** (52 weeks), preserving both date-in-month and day-of-week. The anchor holds from inside the data: the first invite in the log (25 Sep 2025) falls one day after programs gained ERAS access, and the release date sits exactly **28 days** after that access date.

*(The ERAS program-access date — 4th Wednesday of September, 24 Sep 2025 — is external context, not stated in the sheet. The invite log corroborates it: zero invites before 25 Sep, then an immediate trickle.)*

| Event | 2025–26 | → 2026–27 projection |
|---|---|---|
| Programs gain ERAS access | Wed 24 Sep 2025 | **Wed 23 Sep 2026** |
| First trickle of invites | Thu 25 Sep 2025 | **Thu 24 Sep 2026** |
| **Universal release date** | **Wed 22 Oct 2025** | **▶ Wed 21 Oct 2026** |
| 75% of invites landed | Mon 27 Oct 2025 | Mon 26 Oct 2026 |
| 90% landed | Wed 5 Nov 2025 | Wed 4 Nov 2026 |
| 95% landed | Fri 21 Nov 2025 | Fri 20 Nov 2026 |

**Primary projection: Wednesday 21 October 2026.**
**Alternative: Wednesday 28 October 2026**, if APDS anchors to "4th Wednesday of October" rather than "program access + 28 days." Both rules produced 22 Oct in 2025, so last year cannot discriminate between them. **Confirm against the APDS statement and the AAMC ERAS 2027 calendar before making plans around this.**

The sheet's own countdown cells corroborate the back half: ROL deadline ≈ 4 Mar 2027, Match Week from 16 Mar, **Match Day 20 Mar 2027**.

### 13.2 Expected season shape

| Week of | Expected share |
|---|---|
| 21 Sep 2026 | ~1.5% |
| 28 Sep | ~4.7% |
| 5 Oct | ~4.8% |
| 12 Oct | ~12.6% |
| **19 Oct** | **~51.5%** |
| 26 Oct | ~12.7% |
| 2 Nov | ~4.9% |
| 9 Nov onward | ~5% total |

### 13.3 Ten falsifiable predictions

| # | Prediction | Basis |
|---|---|---|
| P1 | Busiest single day > 25% of all reported invites | 35.4% in 25-26 |
| P2 | That day is a Wednesday in the 3rd–4th week of October | Wed held 45.4% |
| P3 | ≥50% of invites land within 7 days of the release date | 50% by release day |
| P4 | Signalled share of invitees between **55–62%** | 53.8%, trending up |
| P5 | Signal share at most-sought vs low-demand programs differs by ≥15 pp | 24.8 pp, bootstrap CI excludes 0 |
| P6 | Median Step 2 CK of invitees 255–260 | 255 in 25-26; ~1 pt/yr national drift |
| P7 | IMGs ≥50% of prelim invites, <15% of categorical | 78% vs 8% |
| P8 | Geo-preference share stays in 70–78% | 73.9% |
| P9 | LOI response rate <50%, personalised <35% | 34% / 30% |
| P10 | Interview-drop activity peaks 2–4 weeks after release | drops clustered late Oct–Nov |

**P4 reasoning:** signal count stays at 15 (the 26-27 `APPLICANT INFO` tab already tracks *"II by Signals x/15"*). With the mechanism unchanged and applicants a year better at concentrating signals, the signalled share of invitees should hold or drift up.

---

## 13. What this means if you're applying

Ordered by leverage, and stated at the confidence the data supports.

0. **Get the interview count up; nothing else comes close.** 9.1× the odds of a categorical match
   per five extra interviews, and 91% of applicants with ten or more matched categorical against 0%
   of those with two or fewer. Every other item on this list is instrumental to this one.

1. **Spend signals where you'd actually go, not where you'd be flattered to go.** The signal is worth roughly 3.9× on the odds. But 75% of invitees at the most-sought programs signalled, so an unsignalled application there is competing for a quarter of the seats. Signals to reach programs are mostly consumed by the crowd; signals to programs that fit your profile convert at 4.5× odds because almost nobody else spends one there.

2. **Do not spend a signal on your home program.** Home-program invitees signalled at exactly the base rate (50.0% vs 54.2%, p=0.64) with no visible penalty. That's a free signal.

2b. **Stop adding programs past about sixty.** Marginal return falls from +1.9 interviews per ten
   applications to zero, and the 120+ group ends with fewer interviews than the 40–59 group. The fee
   money and supplemental-essay hours are better spent on applications already sent. *(Confounded —
   see §1c for what does and does not follow.)*

2c. **Treat publications as the lowest-yield item on your list.** Holding score, pathway, AOA and
   class rank constant, publication count has no measurable effect on interview count (IRR 1.01,
   p = 0.85). This is the clearest gap between what applicants optimise and what the data rewards.

3. **Clear the score filter, then stop optimising it.** 24% of invitees were below 250 and score has no measurable relationship to anything downstream. Past the gate, effort belongs elsewhere. *(Caveat §11.8: effects under ~10 pp are invisible here.)*

4. **An away rotation buys something a signal cannot** — an exemption from score screening at that program, per a PD directly. If your score is your weak point, this is the targeted fix.

5. **Geography is a real filter, not a soft preference.** 74% of invitees had geographic ties to the program that invited them. Make the tie explicit and specific.

6. **Do not read the 21st of October as a verdict.** Signal share peaks that week (60.6%) and falls to 27.8% by mid-November. Unsignalled invites arrive off-peak. The season has a long tail through January.

7. **Send the LOI, expect nothing.** 34% response rate, 30% personalised. Cheap to send, near-useless as a signal about your standing — and a PD in the sheet says a non-response means nothing at all.

8. **If you're a Non-US IMG:** the data says signals convert near zero for you and the prelim track is where 78% of your invites live. The successful counterexample in the sheet — 6–7 categorical invites at 0/15 signals — ran on research volume and mentors making calls. Weight your effort accordingly.

---

## Appendix A — Full tab inventory

| Tab | Rows × Cols | Filled cells | Content |
|---|---|---|---|
| `PCPD Q&A 25-26` | 276 × 8 | 634 | 88 KB PD/PC answers ← qualitative goldmine |
| `II by Date 25-26` | 162 × 122 | 930 | **1,154 invite records ← primary dataset** |
| `IMG Chat` | 105 × 7 | 332 | IMG outcomes, signal yields |
| `Program Details 26 - 27` | 121 × 11 | 908 | EMR, call, chair letter, custom PS |
| `RANK SPOTS 25-26` | 353 × 25 | 2,213 | Rank distributions, 348 programs |
| `2nd Look 26-27` | 347 × 10 | 1,045 | Empty shell |
| `II by Program 25-26` | 341 × 27 | 1,041 | Invite drops (sparse) |
| `HomeAway Student Impressions 26` | 339 × 5 | 1,009 | Rotation impressions |
| `II by Date 25-26 IMGs` | 85 × 8 | 543 | 84 structured IMG records |
| `IV IMPRESSIONS 26-27` | 407 × 16 | 417 | Empty shell |
| `LOI Outcomes` | 178 × 28 | 510 | 69 LOI records |
| `IIs dropped 25-26` | 127 × 45 | 327 | Interview churn |
| `Comm 25-26` | 20 × 17 | 217 | **Application denominators ← critical** |
| `Rejection & Waitlist 25-26` | 23 × 9 | 121 | 20 rejections |
| `Letter of intent response` | 31 × 7 | 74 | 40 LOI outcomes |
| `APPLICANT INFO 26-27` | 10 × 28 | 105 | 3 entries |
| *14 further tabs* | — | <100 each | Chat, swaps, admin |

## Appendix B — Reproducibility

Analysis in Python 3.14 (pandas 3.0.5, numpy 2.5.1, scipy 1.18.0, statsmodels 0.14.6). Scripts in the session scratchpad:

| Script | Purpose |
|---|---|
| `parse_ii.py` | 2025–26 free text → 1,154 structured invite records |
| `parse_2425.py` | 2024–25 applicant tab → 361 records; recovers the date-coerced signal column |
| `analyze.py` | Composition, base rates, signal lift, timing |
| `analyze2.py` | Program tiers, geography, IMG, letters of intent |
| `analyze3.py` | Clustering, bootstrap, multiplicity, 2026–27 projection |
| `combined.py` | Both cycles: yield curve, outcome models, two-year comparison |
| `img_analysis.py` | Non-US IMG stratified analysis |
| `export_site.py` | Emits `assets/site.json`, which every published figure reads |

All statistics use weighted records (tallies expanded to one row per invite). Regression uses program-clustered robust covariance. Bootstrap = 4,000 cluster resamples, seed 7.

## Appendix C — Prior-year sheets (for true multi-year trending)

Extracted from `Overview` hyperlinks. Running `parse_ii.py` against these would give the genuine multi-cycle trend this workbook alone cannot support — in particular whether the signalled share of invitees is rising and whether the release-date concentration is tightening.

| Cycle | URL |
|---|---|
| 2025–2026 | `docs.google.com/spreadsheets/d/18qkHZAbRTsxKbYtN1dSMu6mm5UR3sUPU9MeTA3XL6wk` |
| 2024–2025 | **analysed in this report** · `docs.google.com/spreadsheets/d/1nD9SDVGHxjtRqAnWPBNiKoyO5jARMZl7jKnb0x-MckA` |
| 2023–2024 | `docs.google.com/spreadsheets/d/1MYqD0QtWvg3a65k8O6Qgm3VCDWTFCvfhPB33y9dxbLU` |
| 2022–2023 | `docs.google.com/spreadsheets/d/1Y_ByeeoYaIlwvM9Kw8Ot6b1H7Sellu9BZz_hg4OG5lo` |
| 2021–2022 | `docs.google.com/spreadsheets/d/1xxvKFojfuFDhSRaknkuOVjA1mHU3xI46Vq7EUh27-pw` |
| 2020–2021 | `docs.google.com/spreadsheets/d/17CG5rEzZtH9pS74U76VX9cR51bu9TYiOYLHKhbxq1bY` |
| 2019–2020 | `docs.google.com/spreadsheets/d/1dcR0BLeiQmfcvMMUy7Ypg0YnBCjpLMSRGwguGE4h9w8` |

**Caveat:** the invite-record format (`+/- home, +/- away, +/- signal...`) postdates the introduction of programme signalling. Older sheets will need a different parser, and pre-signalling cycles have no signal field at all.

## Appendix D — What the 26-27 sheet should collect

Gaps that limited this analysis, in order of what they'd unlock:

1. **A stable per-applicant ID** (even a nickname) on every invite record. Currently invites cannot be linked to an applicant, which makes applicant-level clustering impossible and blocks any yield-per-applicant analysis. **Highest value by far** — the `APPLICANT INFO` tab already has a `Nickname/Alias` column that the invite log doesn't reference.
2. **Application counts alongside invite counts.** `APPLICANT INFO` has `# Applications` and `YIELD %` columns — if they were actually filled in, the denominator reconstruction in §0 would be unnecessary.
3. **Signals spent vs. signals converted**, per applicant. The `II by Signals x/15` column exists; nobody fills it. This would turn the entire signal analysis from inference into direct measurement.
4. **Gold vs silver signal distinction**, if the tier structure persists.
5. **Rejections logged with the same discipline as invites.** 20 rejection records against 1,154 invites means the negative case is essentially unobserved — the root cause of the denominator problem.
6. **Explicit `Cat`/`Prelim` tagging on every invite,** as the IMG tab already does well. Currently inferred from free text.

---

*Analysis of a voluntary, self-reported dataset covering roughly 5–6% of national General Surgery interview volume. Findings describe reported invites and should not be read as national rates. Where estimates rest on assumptions, the assumptions are stated inline.*
