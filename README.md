# Redrob Hackathon — Candidate Ranker

A hybrid ranking system for the Intelligent Candidate Discovery & Ranking
Challenge: rank 100,000 candidates against a Senior AI Engineer JD the way
a sharp recruiter would — by reading career history for substance, not by
counting skill keywords.

**TL;DR result:** the top 100 contains zero honeypots, zero consulting-only
candidates, zero pure-research-only candidates, and correctly surfaces
"Tier 5" candidates (e.g. `Recommendation Systems Engineer`, `Search
Engineer`) whose titles carry no AI buzzword but whose career history is
exactly what the JD asks for — see [Results](#results) below.

## Quick start

```bash
pip install -r requirements.txt   # only needed for tests/EDA; rank.py is stdlib-only
gunzip -k data/candidates.jsonl.gz   # produces data/candidates.jsonl (~465MB)
python rank.py --candidates data/candidates.jsonl.gz --out outputs/submission.csv --reference-date 2026-06-24
python scripts/validate_submission.py outputs/submission.csv
```

`rank.py` reads `.jsonl` or `.jsonl.gz` transparently — gunzipping first is
optional. On a single CPU core, the full 100K-candidate pool scores in
**~50–60 seconds** and peaks at **~1.8GB RAM**, comfortably inside the
hackathon's 5-minute / 16GB / CPU-only / no-network budget (see
[Compute](#compute-and-reproducibility) below for the measured numbers).

The `--reference-date` flag pins "today" for recency-based scoring
(`last_active_date` recency, etc.) so the output is exactly reproducible
across runs and machines. Omit it to use the actual current date.

## Why this approach

The JD (`data/job_description.md`) is unusually explicit about what it
*doesn't* want, and the dataset (`data/candidates.jsonl.gz`) is built with
deliberate traps that punish naive approaches. Two things shaped every
design decision here:

1. **The JD says the trap outright:** *"The 'right answer' to this JD is
   not 'find candidates whose skills section contains the most AI
   keywords.' ... A candidate who has all the AI keywords listed as skills
   but whose title is 'Marketing Manager' is not a fit, no matter how
   perfect their skill list looks."*
2. **`data/sample_submission.csv` (the format reference) is itself a
   keyword-stuffing ranker** — its top picks are HR Managers and Content
   Writers ranked by raw "AI core skills" count. We checked: this is
   exactly the failure mode the JD warns about, sitting right in the
   bundle as a cautionary example.

So the core engineering problem isn't "build an ML ranker" — it's **build
a ranker that reads career narrative as evidence, and treats listed skills
as claims that need corroboration**, while staying inside a CPU-only,
5-minute, no-network compute budget that rules out calling an LLM per
candidate.

## Architecture

```
candidates.jsonl(.gz)
        │
        ▼
┌─────────────────────┐
│  features.py         │  Extract structured signals from each candidate:
│  extract_features()  │   - skill trust score (endorsements × duration,
│                       │     not raw count) per skill bucket
│                       │   - career_relevance_score: regex/term-based read
│                       │     of career_history TEXT (not title) for
│                       │     production/eval/scale/domain language
│                       │   - trap flags: honeypot, consulting-only,
│                       │     title-chaser, pure-research
│                       │   - location/notice/education signals
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  scoring.py           │  base_fit_score = weighted blend of:
│  score_candidate()    │    skill_substance × corroboration_factor
│                        │    + career_relevance + experience_fit
│                        │    + seniority_authenticity + location + education
│                        │
│                        │  final_score = base_fit_score
│                        │    × behavioral_multiplier   (engagement signals)
│                        │    × trap_penalty_multiplier (hard disqualifiers)
└─────────┬──────────────┘
          ▼
┌─────────────────────┐
│  reasoning.py         │  Composes a fact-grounded, varied, rank-consistent
│  generate_reasoning() │  1-2 sentence justification from the SAME boolean/
│                        │  numeric features used in scoring — nothing here
│                        │  is invented; every clause traces to a real field.
└─────────┬──────────────┘
          ▼
   outputs/submission.csv   (validated against scripts/validate_submission.py)
```

### Why three multiplicative layers instead of one big weighted sum

`base_fit_score` answers "how well does this person's skills/career/
experience/location objectively match the role." `behavioral_multiplier`
answers a *different* question the JD asks explicitly: *"a perfect-on-paper
candidate who hasn't logged in for 6 months and has a 5% recruiter
response rate is, for hiring purposes, not actually available. Down-weight
them appropriately."* Keeping it as a multiplier (bounded to
`[0.55, 1.10]`, see `config.py`) means engagement signals can meaningfully
move a candidate but can't single-handedly drag a strong fit below a weak
one, or vice versa — exactly "down-weight," not "override."

`trap_penalty_multiplier` is separate again because the JD's named
disqualifiers (pure-research-only, consulting-only-career, honeypots) are
described as near-categorical rejections, not "slightly less good"
signals — a multiplicative floor (e.g. `0.02` for honeypots) keeps the
score strictly ordered and debuggable while still pushing these firmly out
of any realistic top-100 cut.

### Defeating the central trap: skill–career corroboration

The single most important piece of logic is in `scoring.py`:
`_skill_career_corroboration_factor()`. During EDA (see
`notebooks/eda.py`, and the worked example below) we found candidates like
a 14.5-year Project Manager whose `skills` array includes "Recommendation
Systems," "Embeddings," and "Vector Search" — each with plausible-looking
endorsement counts and multi-month durations — while every entry in their
`career_history` describes brand design, mechanical engineering, sales,
and customer support. Trust-weighting skills by endorsements/duration
(which we also do) does **not** catch this, because the numbers on the
skill entries themselves look reasonable in isolation.

The fix: skill substance is only counted at full value if
`career_relevance_score` (computed purely from `career_history`
descriptions, never from the `skills` array) clears a threshold. If a
candidate's actual job history shows no corroborating domain signal, their
claimed core skills are discounted to as little as 25% of their
trust-weighted value. This single check moved that Project Manager
candidate out of the top 100 entirely.

### Catching the "Tier 5" plain-language candidates

The flip side of the same idea: `career_relevance_score` is computed from
**free-text career history**, scanning for domain terms (retrieval,
ranking, recommendation, embeddings, hybrid search, click-through, etc.)
and production/evaluation/scale language (shipped, NDCG, p95, QPS), and is
completely independent of the `current_title` string. This is what lets a
`Search Engineer` or `Recommendation Systems Engineer` — titles that
contain zero AI/ML buzzwords — score competitively with an explicitly
titled `Senior AI Engineer`, provided their actual work history backs it
up. In the final run, both title families are well represented in the top
100 (see [Results](#results)).

## Honeypot, consulting-only, and pure-research detection

All three are named explicitly in `data/job_description.md` and
`data/submission_spec.md`. Implementation and the EDA that grounds each
threshold lives in `src/features.py` / `src/config.py`; the short version:

- **Honeypots**: "expert" or "advanced" proficiency claimed with
  implausibly low `duration_months` (≤3 / ≤1 months respectively), or an
  implausibly broad spread of simultaneous "expert" skills (≥8). During
  EDA, every candidate matching the tightest version of this signature had
  an off-domain title (HR Manager, Accountant, Civil Engineer, etc.),
  which is consistent with this being a deliberately planted signature
  rather than a real "fast learner" pattern.
- **Consulting-only**: every company in `career_history` (current company
  included) is in `{TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini,
  HCL}`. Checked against the **full** career, not just the current
  employer — the JD explicitly says someone *currently* at a consulting
  firm with prior product-company experience is fine.
- **Pure research**: every role's title matches research-only patterns
  (Research Scientist, Research Intern, Postdoc, etc.) **and** no
  production-deployment language appears anywhere in the career text. Both
  conditions are required so we don't penalize a Research Scientist who
  later shipped something to production.

All three apply a hard multiplicative penalty (see `config.py`) rather
than removing the candidate outright, which keeps every score strictly
ordered and auditable — but in practice none of these survive into the
top 100 in the final run.

## Results

Running `python rank.py` on the full 100K pool with `--reference-date
2026-06-24` produces a top 100 with:

| Check | Result |
|---|---|
| Honeypot rate in top 100 | **0%** (limit: ≤10%) |
| Consulting-only-career candidates in top 100 | **0** |
| Pure-research-only candidates in top 100 | **0** |
| Off-domain titles (HR/Accounting/Sales/etc.) in top 100 | **0** |
| Title-chaser candidates in top 100 | 4 (present despite the tenure penalty because their skill/career substance is exceptional — JD frames title-chasing as a real concern, not a categorical reject) |
| "Tier 5" plain-language titles in top 100 (Search Engineer, Recommendation Systems Engineer) | 24 |
| Distinct AI/ML-flavored titles in top 100 | 17 title variants, from Junior ML Engineer through Staff Machine Learning Engineer |
| `scripts/validate_submission.py` | **Passes** |
| Wall-clock runtime (full 100K) | **~51–61s** (measured across multiple runs) |
| Peak RSS | **~1.8 GB** |

## Reasoning column design

`submission_spec.md` Section 3 samples 10 rows at Stage 4 and checks for
six things: specific facts, JD connection, honest concerns, no
hallucination, variation across rows, and rank consistency. Calling an LLM
per candidate to write these is explicitly disallowed by the compute
budget (100K candidates, no network, 5 minutes), so `reasoning.py` builds
each justification from a **pool of fact-gated clause templates** —
every clause is conditioned on a boolean or numeric feature computed
directly from that candidate's own profile (years of experience, named
skill count, notice period, response rate, etc.), with several phrasing
variants per clause selected via a seeded RNG so the same underlying fact
isn't always expressed identically across candidates.

One real bug we caught and fixed during development: an earlier version
branched the reasoning's tone on **rank position** (e.g. "if rank > 70,
lead with a concern"). That's wrong — rank position within an
already-curated top 100 doesn't mean "this candidate is weak," it just
means "99 people scored higher." We had cases where a maxed-out-on-substance
candidate landed near rank 90 (because the other 89 were even stronger)
and got "marginal/weak fit" text that flatly contradicted their own
profile — exactly the kind of rank-inconsistency the spec checks for, just
inverted. Fixed by branching on **actual strengths/concerns content**
instead of rank number; see the git history for the before/after.

## Compute and reproducibility

`rank.py` and everything in `src/` use **only the Python standard
library** (`json`, `csv`, `gzip`, `re`, `datetime`, `dataclasses`,
`argparse`). This was a deliberate choice, not an oversight: it trivially
satisfies the "CPU-only, no network, 5-minute, 16GB" constraint with no
embedding-model download, no vector-DB process, and no version-drift risk
at Stage 3 reproduction. `requirements.txt` only lists packages needed for
the optional EDA script and the test suite.

Single command to reproduce the submission CSV from the candidate file
(also declared in `submission_metadata.yaml`):

```bash
python rank.py --candidates ./data/candidates.jsonl.gz --out ./outputs/submission.csv --reference-date 2026-06-24
```

## Tests

```bash
python -m unittest tests/test_traps.py -v
```

12 tests pin down expected behavior for every named trap (honeypot,
consulting-only with the "but prior product company is fine" exception,
title-chaser, the skill–career corroboration gate, pure-research with the
"unless they shipped something" exception) plus a monotonicity check that
behavioral signals only move the multiplier layer, never the base fit
score. These run in milliseconds against small synthetic fixtures, not the
real dataset — see `tests/test_traps.py`.

## EDA

`notebooks/eda.py` is the record of the actual data exploration that
produced the thresholds and skill taxonomy in `src/config.py` — title
distribution, the Tier-5 trap discovery, the honeypot signature check, the
title-chaser/consulting-only prevalence within the AI-titled cohort, and
the skill-vocabulary frequency bands that motivated `CORE_SKILLS` /
`NICE_TO_HAVE_SKILLS` / `CV_SPEECH_SKILLS`. Run with:

```bash
python notebooks/eda.py
```

## Sandbox / demo

`sandbox/app.py` is a Streamlit app implementing the same pipeline against
a bundled 50-candidate sample (`data/sample_candidates.json`) — run
locally with `streamlit run sandbox/app.py`, or deploy on Streamlit
Community Cloud (free tier) pointing at this file. It lets you upload your
own small JSON/JSONL sample, runs the full scoring pipeline in-browser, and
offers a submission-format CSV download plus a score breakdown for the
top candidate.

## Repository layout

```
.
├── rank.py                          # CLI entry point — the reproduce command
├── src/
│   ├── config.py                    # All domain-knowledge constants, with rationale
│   ├── features.py                  # Raw candidate JSON -> structured CandidateFeatures
│   ├── scoring.py                    # CandidateFeatures -> final score (3 multiplicative layers)
│   └── reasoning.py                  # Fact-grounded, varied reasoning text generation
├── tests/test_traps.py              # 12 unit tests against synthetic fixtures
├── notebooks/eda.py                  # The exploration that grounds config.py's constants
├── sandbox/app.py                    # Streamlit demo app (sandbox link requirement)
├── scripts/validate_submission.py    # Organizer-provided format validator
├── data/                             # JD, schema, signals doc, gzipped candidate pool, 50-sample
├── outputs/submission.csv            # Final ranked top-100 output
└── submission_metadata.yaml          # Portal metadata mirror
```

## Honest limitations

- `career_relevance_score` is a regex/term-density heuristic over career
  text, not a learned semantic model — it can miss genuinely novel
  phrasings of the same work and can be fooled by someone who happens to
  use the right vocabulary without real substance (though the
  corroboration gate against the *skills* array makes pure vocabulary
  stuffing in career text alone less effective, since skill claims still
  need their own trust-weighting to count).
- `is_pure_research` and `current_role_codes` are inferred from title and
  free-text description patterns, not a ground-truth "has this person
  written code in 18 months" field — the dataset doesn't expose one, so
  this is the best available proxy and is documented as such in
  `config.py`.
- Honeypot detection catches the specific "expert-with-near-zero-duration"
  signature we found during EDA; it is not guaranteed to catch every one
  of the ~80 honeypots described in the README, only the subset matching
  patterns we could verify empirically in the released data.
