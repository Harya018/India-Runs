"""
Scoring: turns CandidateFeatures into a single ranking score.

Design: base_fit_score (weighted combination of skill/career/experience/
seniority/location/education) is multiplied by a behavioral_multiplier
(derived from Redrob platform engagement signals) and then by a
trap_penalty_multiplier (honeypots, consulting-only, pure-research,
cv/speech-only-without-NLP). This keeps each layer interpretable and
independently auditable -- you can always ask "why this score" and trace
it to one of these three numbers.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass

from . import config
from .features import CandidateFeatures


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    base_fit_score: float
    behavioral_multiplier: float
    trap_penalty_multiplier: float
    features: CandidateFeatures


def _experience_fit(years: float) -> float:
    """5-9 years is the JD's band, explicitly described as 'a range, not a
    requirement' -- so this is a smooth curve, not a step function. Peaks
    in-band, decays gently outside it (faster on the low side, since the
    JD's disqualifiers skew toward 'too junior for this scope' more than
    'overqualified')."""
    lo, hi = 5.0, 9.0
    if lo <= years <= hi:
        return 1.0
    if years < lo:
        # linear falloff below the band; by 2 years experience -> ~0.35
        return max(0.0, 1.0 - (lo - years) * 0.22)
    # gentle falloff above the band; by 16 years -> ~0.55
    return max(0.35, 1.0 - (years - hi) * 0.065)


def _tenure_stability_factor(features: CandidateFeatures) -> float:
    if features.is_title_chaser:
        return 0.55
    if features.avg_tenure_months >= 24:
        return 1.0
    if features.avg_tenure_months >= 18:
        return 0.85
    return 0.7


def _seniority_authenticity_score(features: CandidateFeatures) -> float:
    if features.current_role_is_management_only and not features.current_role_codes:
        return 0.35
    if features.current_role_is_management_only:
        return 0.7
    return 1.0


def _days_since(date_str: str, reference: date) -> float:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max((reference - d).days, 0)
    except (ValueError, TypeError):
        return 9999


def behavioral_multiplier(features: CandidateFeatures, reference_date: date) -> float:
    """
    Maps Redrob engagement signals to a multiplier in
    [BEHAVIORAL_MULTIPLIER_MIN, BEHAVIORAL_MULTIPLIER_MAX].

    Directly implements the JD's instruction: "a perfect-on-paper candidate
    who hasn't logged in for 6 months and has a 5% recruiter response rate
    is, for hiring purposes, not actually available. Down-weight them
    appropriately." -- but as a multiplier with a floor, not a hard zero,
    since "down-weight" != "disqualify".
    """
    days_inactive = _days_since(features.last_active_date, reference_date)
    recency_factor = (
        1.0 if days_inactive <= 14 else
        0.9 if days_inactive <= 30 else
        0.75 if days_inactive <= 90 else
        0.55 if days_inactive <= 180 else
        0.4
    )

    response_factor = 0.5 + 0.5 * min(features.recruiter_response_rate, 1.0)

    open_to_work_factor = 1.0 if features.open_to_work else 0.85

    completeness_factor = 0.85 + 0.15 * min(features.profile_completeness / 100, 1.0)

    interview_factor = (
        1.0 if features.interview_completion_rate >= 0.6
        else 0.9 if features.interview_completion_rate >= 0.3
        else 0.95 if features.interview_completion_rate == 0  # no signal yet; neutral
        else 0.8
    )

    verification_factor = 0.92 + 0.04 * features.verified_email + 0.04 * features.verified_phone

    raw = (
        recency_factor * 0.32
        + response_factor * 0.28
        + open_to_work_factor * 0.14
        + completeness_factor * 0.12
        + interview_factor * 0.08
        + verification_factor * 0.06
    )
    # raw is a weighted blend of factors already centered near [0.4, 1.0];
    # rescale into the configured multiplier band.
    lo, hi = config.BEHAVIORAL_MULTIPLIER_MIN, config.BEHAVIORAL_MULTIPLIER_MAX
    scaled = lo + (raw - 0.4) / (1.0 - 0.4) * (hi - lo)
    return max(lo, min(hi, scaled))


def trap_penalty_multiplier(features: CandidateFeatures) -> float:
    if features.is_honeypot:
        return config.HONEYPOT_PENALTY_MULTIPLIER
    if features.is_pure_research:
        return config.PURE_RESEARCH_PENALTY_MULTIPLIER
    if features.is_consulting_only:
        return config.CONSULTING_ONLY_PENALTY_MULTIPLIER
    if features.cv_speech_trust_score > 0.4 and not features.has_nlp_or_ir_skill:
        return config.CV_SPEECH_ONLY_PENALTY_MULTIPLIER
    return 1.0


def _skill_career_corroboration_factor(features: CandidateFeatures) -> float:
    """
    The JD's central instruction: 'a candidate who has all the AI keywords
    listed as skills but whose title is "Marketing Manager" is not a fit,
    no matter how perfect their skill list looks.' Trust-weighting
    (endorsements x duration) alone does NOT catch this -- a side-project
    dabbler's skill entries can carry plausible-looking endorsement/duration
    numbers while their actual career_history is entirely unrelated work.

    This factor discounts claimed skill substance when the career
    narrative shows no corroborating domain signal: skills are only worth
    their full trust-weighted value if the person's actual job history
    backs them up. A high core-skill score with near-zero career_relevance
    is the textbook 'AI hobbyist with an unrelated day job' profile the JD
    explicitly disqualifies.
    """
    if features.career_relevance_score >= 0.35:
        return 1.0  # career history corroborates the skills; no discount
    if features.career_relevance_score >= 0.15:
        return 0.55
    return 0.25  # skills claimed with no domain signal anywhere in career history


def base_fit_score(features: CandidateFeatures) -> float:
    w = config.WEIGHTS

    corroboration = _skill_career_corroboration_factor(features)
    skill_component = corroboration * (
        0.75 * features.core_skill_trust_score
        + 0.25 * features.nice_to_have_trust_score
    )
    experience_component = (
        0.6 * _experience_fit(features.years_of_experience)
        + 0.4 * _tenure_stability_factor(features)
    )
    seniority_component = _seniority_authenticity_score(features)

    score = (
        w["skill_substance"] * skill_component
        + w["career_relevance"] * features.career_relevance_score
        + w["experience_fit"] * experience_component
        + w["seniority_authenticity"] * seniority_component
        + w["location_logistics"] * features.location_fit_score
        + w["education"] * features.education_score
    )
    return max(0.0, min(1.0, score))


def score_candidate(
    features: CandidateFeatures, reference_date: date
) -> ScoredCandidate:
    base = base_fit_score(features)
    behavior_mult = behavioral_multiplier(features, reference_date)
    trap_mult = trap_penalty_multiplier(features)
    final = base * behavior_mult * trap_mult
    return ScoredCandidate(
        candidate_id=features.candidate_id,
        score=final,
        base_fit_score=base,
        behavioral_multiplier=behavior_mult,
        trap_penalty_multiplier=trap_mult,
        features=features,
    )
