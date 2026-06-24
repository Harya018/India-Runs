# Development Log

This documents the real debugging that happened while building this
ranker, kept separate from the README so the README stays focused on
"how it works" while this stays focused on "what we found wrong and how
we fixed it." Both bugs were caught by actually running the ranker against
real candidates and inspecting individual outputs — not by code review
alone.

## Bug 1: trust-weighted skills don't defeat "populated" keyword stuffing

**How it was found:** after the first working version of `scoring.py`,
we ran `rank.py` against a 50-candidate sample and manually inspected the
top 10. Rank #8 was a 14.5-year "Project Manager" at Wipro with a score of
0.46 and a reasoning string claiming "7 core IR/retrieval skills backed by
real endorsements and tenure." That set off an alarm — a Project Manager
shouldn't plausibly have 7 legitimate core retrieval skills.

Pulling the full candidate record (`CAND_0000021`) confirmed the problem:

```
"headline": "Project Manager | AI enthusiast | Building with LLMs",
"summary": "...I've been taking online courses on RAG and vector
            databases, experimenting with LangChain and the OpenAI API
            for side projects..."
"skills": [
  {"name": "Recommendation Systems", "proficiency": "advanced", "endorsements": 3, "duration_months": 13},
  {"name": "Embeddings", "proficiency": "advanced", "endorsements": 4, "duration_months": 18},
  {"name": "Vector Search", "proficiency": "intermediate", "endorsements": 3, "duration_months": 13},
  ...
]
```

Every `career_history` entry described brand design, mechanical
engineering, sales, and customer support — nothing related to ML/AI at
all. But the skill entries had *plausible-looking* endorsement counts
(3-4) and durations (13-18 months) — exactly what you'd expect from
someone genuinely dabbling in online courses and side projects, which is
exactly what the candidate's own summary says they're doing.

**Why the existing trust-weighting didn't catch it:** our skill scoring
(`_skill_trust_score()` in `features.py`) weights by endorsements ×
duration specifically to defeat *empty* keyword stuffing (0 endorsements,
0 months). This candidate's numbers weren't empty — they were small but
real, consistent with genuine hobbyist exposure. Trust-weighting alone
treats "genuine hobbyist dabbling" and "genuine production experience" as
differing only in degree, when the JD treats them as categorically
different things.

**The fix:** added `_skill_career_corroboration_factor()` in
`scoring.py`. It gates the entire skill-substance component by
`career_relevance_score` — which is computed independently, purely from
`career_history` description text, never from the `skills` array. If a
candidate's actual job history shows no corroborating domain signal
(`career_relevance_score < 0.15`), their claimed skill substance is
discounted to 25% of its trust-weighted value, regardless of how
reasonable the endorsement/duration numbers look in isolation.

**Verified impact:** `CAND_0000021`'s rank moved from #8 (score 0.46) to
outside the top 100 entirely once the full 100K pool was scored. The
"Recommendation Systems Engineer" Tier-5 candidate (`CAND_0000031`), whose
skills ARE corroborated by career history, stayed at the top of the
ranking throughout — confirming the fix discriminates on the right axis
(corroboration), not just on skill-list size.

A unit test for exactly this case lives in `tests/test_traps.py`:
`TestSkillCareerCorroboration.test_ai_skills_with_unrelated_career_history_penalized`.

## Bug 2: reasoning text contradicted the candidate's own score

**How it was found:** after running the full ranker end-to-end and
validating format, we did a manual read-through of ranks 88-100 in the
output (not just the top 10 — the spec samples 10 *random* rows at Stage
4, so the tail matters as much as the head). Three entries stood out:

```
91  CAND_0033179  0.8689  "...only marginal signal for this JD; included as lower-confidence filler."
96  CAND_0065195  0.8663  "...only marginal signal for this JD; included as lower-confidence filler."
98  CAND_0043860  0.8647  "...only marginal signal for this JD; included as lower-confidence filler."
```

A score of 0.87 is near the top of an already-curated top-100 — calling it
"marginal" and "lower-confidence filler" is actively wrong, and exactly
the kind of "reasoning that contradicts the rank" failure mode
`submission_spec.md` Section 3 explicitly checks for (just inverted: we
expected to find *overly generous* text on weak candidates, not overly
harsh text on strong ones).

**Root cause:** `generate_reasoning()`'s original implementation branched
its composition strategy on **rank number** (`if rank <= 30: ... elif
rank <= 70: ... else: ...`), with the `else` branch (rank > 70) defaulting
to "marginal/weak fit" language whenever the `concerns` list happened to
be empty — without checking whether `strengths` was actually populated.
Debugging confirmed it: pulling `_strength_clauses()` /
`_concern_clauses()` directly for `CAND_0033179` showed 5 real strengths
and zero concerns — a clean profile that simply had 90 *even stronger*
peers ahead of it in this particular pool. Rank position in an
already-curated top 100 is not a reliable proxy for "is this candidate
actually weak."

**The fix:** rewrote the branching logic in `reasoning.py` to key off
**actual strengths/concerns content** instead of rank number:
- strengths present, no concerns → state the strengths plainly, no
  manufactured caveat
- both present → state both, honestly, with "however"
- concerns present, no strengths → lead with the limiting factor
- neither → say so explicitly rather than inventing praise or criticism

**Verified impact:** re-ran the full pipeline; all three flagged rows now
read accurately (e.g. rank 91 became *"...career history shows shipped
production systems directly in ranking/retrieval work; 5 core IR/retrieval
skills backed by real endorsements and tenure, not just listed; 6.9 yrs
experience sits in the JD's target band."* — accurate, and consistent with
an 0.87 score). Spot-checked two additional random 10-row samples
afterward; no further rank-inconsistent text found.

## What this means for the "defend your work" interview

Both bugs were found by **running the actual ranker against real data and
reading individual outputs critically** — not by reasoning about the code
in the abstract. We'd point to this as the main evidence that the
engineering here is real: the first bug required recognizing that a
"reasonable-looking" data pattern was actually a trap requiring a new
cross-field check; the second required noticing that our own generated
text contradicted our own scores, which only shows up if you actually read
the output rather than just checking it's well-formed.
