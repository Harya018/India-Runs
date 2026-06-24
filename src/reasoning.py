"""
Reasoning generation for the submission CSV's `reasoning` column.

submission_spec.md Section 3 samples 10 rows at Stage 4 and checks for:
  1. Specific facts (years, title, named skills, signal values)
  2. JD connection (not generic praise)
  3. Honest concerns (acknowledges gaps, doesn't just praise)
  4. No hallucination (every claim traces to the candidate's actual profile)
  5. Variation (not templated across candidates)
  6. Rank consistency (tone matches rank -- no glowing text at rank 95)

To satisfy all six without an LLM call (which is banned at ranking time by
the compute constraints), this module builds reasoning from a *pool* of
fact-grounded clause templates selected based on which features actually
fired for that candidate, then composes them with light grammatical
variation. Every clause is gated on a boolean/value that is computed in
features.py from the candidate's own data -- nothing here is invented.
"""

from __future__ import annotations

import random

from .scoring import ScoredCandidate


def _fmt_years(y: float) -> str:
    return f"{y:.1f}".rstrip("0").rstrip(".") if y == int(y) else f"{y:.1f}"


def _strength_clauses(sc: ScoredCandidate, rng: random.Random) -> list[str]:
    f = sc.features
    clauses = []

    if f.career_relevance_score >= 0.55:
        signals = []
        if f.has_shipped_production_signal:
            signals.append("shipped production systems")
        if f.has_eval_framework_signal:
            signals.append("built evaluation/NDCG-style rigor")
        if f.has_scale_signal:
            signals.append("operated at real scale")
        opener_variants = [
            "career history shows {} directly in ranking/retrieval work",
            "track record includes {} on ranking/retrieval problems",
            "has hands-on history of {} in search/recommendation work",
        ]
        opener = rng.choice(opener_variants)
        if signals:
            clauses.append(opener.format(", ".join(signals)))
        else:
            fallback_variants = [
                "career history maps closely onto ranking/retrieval work",
                "job history is squarely in search/recommendation/retrieval territory",
            ]
            clauses.append(rng.choice(fallback_variants))

    if f.core_skill_trust_score >= 0.5:
        skill_phrasings = [
            "{} core IR/retrieval skills backed by real endorsements and tenure, not just listed",
            "depth on {} core retrieval/ranking skills, evidenced by endorsements and months of actual use",
            "{} core retrieval-relevant skills with credible endorsement/tenure backing",
        ]
        clauses.append(rng.choice(skill_phrasings).format(f.n_core_skills))
    elif f.core_skill_trust_score >= 0.25:
        clauses.append(f"some core retrieval/ranking skill depth ({f.n_core_skills} skills)")

    if 5 <= f.years_of_experience <= 9:
        clauses.append(f"{_fmt_years(f.years_of_experience)} yrs experience sits in the JD's target band")

    if f.recruiter_response_rate >= 0.6:
        clauses.append(f"responsive ({f.recruiter_response_rate:.0%} recruiter response rate)")

    if f.open_to_work and f.notice_period_days <= 30:
        clauses.append(f"open to work with a {f.notice_period_days}-day notice period")

    if f.location.lower().split(",")[0].strip() in ("pune", "noida"):
        clauses.append(f"based in {f.location.split(',')[0]}, the role's preferred location")

    return clauses


def _concern_clauses(sc: ScoredCandidate) -> list[str]:
    f = sc.features
    concerns = []

    if f.is_consulting_only:
        concerns.append("entire career has been at consulting firms, which the JD flags as a fit risk")
    if f.is_pure_research:
        concerns.append("background reads as pure research without visible production deployment")
    if f.is_title_chaser:
        concerns.append(f"short average tenure (~{f.avg_tenure_months:.0f} months/role) across {f.n_roles} roles")
    if f.current_role_is_management_only and not f.current_role_codes:
        concerns.append("current title suggests management focus with limited hands-on coding signal")
    if f.cv_speech_trust_score > 0.4 and not f.has_nlp_or_ir_skill:
        concerns.append("skill set leans CV/speech without visible NLP/IR exposure, a specialization gap per the JD")
    if f.core_skill_trust_score >= 0.3 and f.career_relevance_score < 0.15:
        concerns.append(
            "lists several AI/retrieval skills but career history shows no "
            "corroborating production work in that domain -- reads as "
            "side-project/course exposure layered on an unrelated day job"
        )
    if f.years_of_experience < 5:
        concerns.append(f"only {_fmt_years(f.years_of_experience)} yrs experience, below the JD's 5-9 yr band")
    elif f.years_of_experience > 9:
        concerns.append(f"{_fmt_years(f.years_of_experience)} yrs experience, above the JD's target band")
    if f.notice_period_days > 60:
        concerns.append(f"{f.notice_period_days}-day notice period is longer than the JD's stated preference")
    if not f.open_to_work:
        concerns.append("not currently flagged open to work")
    days_inactive_signal = f.last_active_date
    if sc.behavioral_multiplier < 0.75:
        concerns.append(f"engagement signals are weak (last active {days_inactive_signal}, "
                         f"{f.recruiter_response_rate:.0%} response rate)")
    if f.country.lower() != "india" and f.location_fit_score < 0.4:
        concerns.append(f"based outside India ({f.country}) with no stated relocation signal; JD does not sponsor visas")

    return concerns


def generate_reasoning(sc: ScoredCandidate, rank: int, rng: random.Random) -> str:
    f = sc.features

    if f.is_honeypot:
        reason_text = "; ".join(f.honeypot_reasons[:2])
        return (
            f"Profile shows internal inconsistencies ({reason_text}) consistent "
            f"with a flagged/honeypot record; excluded from serious consideration "
            f"despite surface-level skill listing."
        )

    strengths = _strength_clauses(sc, rng)
    concerns = _concern_clauses(sc)

    opener_pool = [
        f"{f.current_title} ({_fmt_years(f.years_of_experience)} yrs) at {f.current_company}",
        f"{_fmt_years(f.years_of_experience)}-yr {f.current_title}, currently at {f.current_company}",
        f"Currently {f.current_title} at {f.current_company}, {_fmt_years(f.years_of_experience)} yrs experience",
    ]
    opener = rng.choice(opener_pool)

    # Compose based on what's ACTUALLY true for this candidate -- strengths
    # and concerns content, not the rank number. Rank position within an
    # already-curated top-100 is too coarse a proxy for "is this profile
    # actually strong"; branching on rank caused real bug where a
    # maxed-out-on-substance candidate landed near rank 90 (because peers
    # were even stronger) and got "marginal/weak fit" text that flatly
    # contradicted their own profile facts -- exactly the rank-consistency
    # failure the spec checks for, just inverted (good candidate, bad text).
    if strengths and not concerns:
        # Clean profile: lead with strengths, no manufactured caveat.
        body = "; ".join(strengths[:3])
        text = f"{opener}: {body}."
    elif strengths and concerns:
        # Real strengths AND real concerns: state both, honestly.
        body = "; ".join(strengths[:2])
        text = f"{opener}: {body}; however {concerns[0]}."
    elif concerns and not strengths:
        # Genuine weak fit: lead with the limiting factor.
        text = f"{opener}: {concerns[0]}."
        if len(concerns) > 1:
            text = f"{opener}: {concerns[0]}; also {concerns[1]}."
    else:
        # No strong signal either way (rare; e.g. thin profile that still
        # cleared the trap filters) -- describe the situation honestly
        # rather than manufacturing praise or criticism.
        text = (
            f"{opener}: profile clears the JD's hard filters but shows "
            f"limited direct evidence of core retrieval/ranking work; "
            f"included on experience-band and skill-tag fit alone."
        )

    return text
