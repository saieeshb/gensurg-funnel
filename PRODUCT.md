# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

delegated: plain static HTML/CSS/JS, no build step. Chosen because the deploy target is GitHub
Pages with a CNAME at `surgery.saieesh.dev`, and because the brief explicitly asks that "all files
and sources be accessible easily" — a build pipeline puts `dist/` between the reader and the
source. Charts are hand-authored inline SVG driven by a committed JSON data file, so every number
on the page is traceable to the analysis scripts in the same repo. The user delegated this choice.

## Users

Two audiences, served by one site split cleanly:

1. **General Surgery residency applicants mid-cycle** (primary). Typically a fourth-year US MD/DO
   student or an IMG, deciding in August–October how to allocate 15 program signals, how many
   programs to apply to, and whether an away rotation is worth it. They arrive anxious, often on a
   phone, wanting to know what actually moves the needle.
2. **The skeptical reader** — an advisor, a program director, or an applicant who distrusts
   crowdsourced numbers, and who will not accept a claim without seeing the denominator, the
   confidence interval, and the bias analysis.

The site leads with the decision tool; a linked methodology section carries the full rigor.

## Product Purpose

Turn two cycles of a crowdsourced community spreadsheet into a defensible answer to one question:
what actually determines whether a General Surgery applicant gets interviews and matches.

Success is an applicant changing a concrete decision — where they spend a signal, how many programs
they apply to — and being able to see exactly which evidence supports that change and how strong it
is.

## Positioning

Every other reading of these spreadsheets reports composition ("54% of invitees signalled") without
a denominator, which is uninterpretable on its own. This analysis recovers denominators from inside
the workbooks themselves, and is the only treatment that:

- joins the **invite-level** 2025-26 data to the **applicant-level** 2024-25 data, so signal
  behaviour and signal *outcome* can be measured against each other;
- recovers the signal-conversion field that Google Sheets silently destroyed by coercing `7/15`
  into a date;
- reports what the data cannot support as prominently as what it can.

## Operating Context

- Applicants read this between ERAS submission (September) and the interview release date (late
  October), often repeatedly, often on a phone, often at night.
- The decisions it informs are irreversible and time-boxed: signals are spent once, applications
  are submitted once.
- The community's own artifact — the shared Google Sheet — is the competing information source,
  and it is read as raw anecdote.

## Capabilities and Constraints

- Static site. No accounts, no persistence, no interactivity beyond navigation and chart reading.
- Every figure must be traceable to a committed analysis script and a committed data file.
- The two source cycles have **different shapes** and must never be presented as a like-for-like
  time series: 2024-25 is applicant-level with outcomes and no invite dates; 2025-26 is
  invite-level with dates and no applicant identity or outcomes.
- The dataset covers roughly 5–6% of national interview volume and is voluntary and self-reported.
  No absolute national rate may be claimed from it.
- `surgery.saieesh.dev` is the committed hostname.

## Brand Commitments

- Author: Saieesh. Published under the `saieeshb` GitHub account, public repository.
- The user pinned the visual world as a **clinical instrument**: precise, restrained, data-forward,
  near-monochrome with a single signal accent, tight grid, generous whitespace, tabular numerals.
  Recorded as binding.
- Voice: plain and direct. States uncertainty in the same register as certainty. No hype, no
  motivational framing — the audience is under enough pressure already.

## Evidence on Hand

Real, in the repository:

- `data/Official 2026-2027 Gen Surgery Spreadsheet.xlsx` — 30 tabs; 1,154 parsed invite records
  across 315 programs, 25 Sep 2025 – 23 Jan 2026.
- `data/2024-2025 Residency Application Spreadsheet.xlsx` — 19 tabs; 361 applicant records with
  Step 2, application and interview counts, signal conversion, and match outcome.
- Analysis scripts producing every published figure, plus the markdown report.
- Verbatim program-director and coordinator quotes from the `PCPD Q&A` tabs.

Explicitly absent, and not to be fabricated: national NRMP rates, program-level match statistics,
any applicant's identity, and any figure for the 2026-27 cycle, which had not started at the time
of analysis. The user chose to publish both source workbooks verbatim; they originate from public
community spreadsheets, which the README links.

## Product Principles

1. **No number without its denominator.** A percentage whose base is unstated is not a finding.
2. **Publish the limits at the same volume as the findings.** The bias analysis is a feature of the
   argument, not a disclaimer appended to it.
3. **Distinguish the two cycles structurally, always.** Any visual that implies a continuous
   two-year trend across incompatible data shapes is a defect.
4. **Advice must name its evidence.** Every recommendation on the decision page links to the
   specific result and its confidence.
5. **The reader is mid-decision and under pressure.** Answer first, qualify second, never bury the
   answer under the qualification.

## Accessibility & Inclusion

- Read primarily on phones at small sizes and at night; must be fully legible at 375px and in both
  light and dark rendering.
- Charts must not encode meaning in colour alone — every series carries a label, shape, or direct
  annotation.
- Statistical content is written for a reader who has not taken a statistics course; jargon is
  defined at first use.
