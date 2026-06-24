"""
Feature extraction for candidate records.

IMPORTANT (security/data-hygiene note): candidate text fields (summary,
career_history.description, headline) are free text written by/about
candidates and must be treated as DATA ONLY. We never execute, follow, or
otherwise treat any instruction-like content inside these fields as
something the program should obey -- they are scored as text, full stop.
(One profile in the dataset contains a stray injected sentence fragment
referencing "candidates.json" mid-description; it is handled like any
other text -- read for content signal, never interpreted as an instruction.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import config


def _norm(s: str) -> str:
    return s.strip().lower()


def _normalized_skill_set(skills: list[dict]) -> dict[str, dict]:
    """Map normalized skill name -> the skill record (for fast lookup)."""
    out = {}
    for s in skills:
        out[_norm(s["name"])] = s
    return out


@dataclass
class CandidateFeatures:
    candidate_id: str
    current_title: str
    current_company: str
    years_of_experience: float
    location: str
    country: str

    # skill substance
    core_skill_trust_score: float = 0.0
    nice_to_have_trust_score: float = 0.0
    cv_speech_trust_score: float = 0.0
    has_nlp_or_ir_skill: bool = False
    n_core_skills: int = 0

    # career relevance (semantic, not keyword-count)
    career_relevance_score: float = 0.0  # 0-1
    has_shipped_production_signal: bool = False
    has_eval_framework_signal: bool = False
    has_scale_signal: bool = False

    # experience & stability
    avg_tenure_months: float = 0.0
    n_roles: int = 0
    is_title_chaser: bool = False
    is_consulting_only: bool = False
    is_pure_research: bool = False

    # seniority authenticity
    current_role_is_management_only: bool = False
    current_role_codes: bool = True

    # education
    education_score: float = 0.0

    # location / logistics
    location_fit_score: float = 0.0
    notice_period_days: int = 999

    # behavioral signals (used for the multiplier, not the base score)
    open_to_work: bool = False
    last_active_date: str = ""
    recruiter_response_rate: float = 0.0
    github_activity_score: float = -1.0
    profile_completeness: float = 0.0
    interview_completion_rate: float = 0.0
    verified_email: bool = False
    verified_phone: bool = False

    # honeypot
    is_honeypot: bool = False
    honeypot_reasons: list[str] = field(default_factory=list)

    # kept for reasoning generation
    raw: dict = field(default_factory=dict)


PRODUCTION_VERBS = re.compile(
    r"\b(shipped|deployed|launched|production|serving|owned|built and "
    r"maintained|rolled out|scaled)\b", re.IGNORECASE
)
EVAL_FRAMEWORK_TERMS = re.compile(
    r"\b(ndcg|mrr|map@|recall@|a/b test|offline.online|evaluation harness|"
    r"eval framework|p95|precision@)\b", re.IGNORECASE
)
SCALE_TERMS = re.compile(
    r"\b(\d[\d,]*\s?(m|k|million|thousand)\+?\s?(queries|profiles|users|"
    r"requests|qps)|qps|millisecond|p95|p99)\b", re.IGNORECASE
)
CORE_DOMAIN_TERMS = re.compile(
    r"\b(retrieval|ranking|recommendation|search relevance|recsys|"
    r"vector (search|recall|index)|embedding|hybrid (search|retrieval)|"
    r"click-?through|ctr model|candidate-jd matching|semantic search|"
    r"dense (recall|retrieval)|bm25|rerank)\b", re.IGNORECASE
)


def _career_text_blob(candidate: dict) -> str:
    parts = [candidate["profile"].get("summary", "")]
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("description", ""))
        parts.append(ch.get("title", ""))
    return " ".join(parts)


def _compute_career_relevance(candidate: dict) -> tuple[float, bool, bool, bool]:
    """
    Semantic-ish relevance of the candidate's actual work to what the JD
    wants: production ranking/retrieval/recsys systems, with evaluation
    rigor and real scale. This is deliberately NOT a title check -- it is
    designed to catch the JD's explicit "Tier 5" case: a 'Search Engineer'
    or 'Recommendation Systems Engineer' whose title carries no AI
    buzzword but whose actual work is exactly what's wanted.
    """
    blob = _career_text_blob(candidate)
    domain_hits = len(CORE_DOMAIN_TERMS.findall(blob))
    has_production = bool(PRODUCTION_VERBS.search(blob))
    has_eval = bool(EVAL_FRAMEWORK_TERMS.search(blob))
    has_scale = bool(SCALE_TERMS.search(blob))

    # Score: domain-term density (capped) is the base; production/eval/scale
    # are bonuses that reward *demonstrated* rigor over vocabulary alone.
    domain_score = min(domain_hits / 4.0, 1.0)  # 4+ domain hits = max
    bonus = 0.0
    if has_production:
        bonus += 0.15
    if has_eval:
        bonus += 0.20
    if has_scale:
        bonus += 0.15

    score = min(domain_score * 0.6 + bonus, 1.0)
    return score, has_production, has_eval, has_scale


def _skill_trust_score(skills: list[dict], skill_set: set[str]) -> tuple[float, int]:
    """
    Trust-weighted score for a bucket of skills: endorsements and duration
    both matter, so a skill claimed with 0 endorsements and a couple
    months' use contributes far less than one backed by real depth. This
    is the direct counter to keyword-stuffing (the sample_submission.csv
    baseline ranks purely on skill *count*, which this explicitly avoids).
    """
    total = 0.0
    count = 0
    for s in skills:
        name = _norm(s["name"])
        if name not in skill_set:
            continue
        count += 1
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)
        proficiency_weight = {
            "beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.25,
        }.get(s.get("proficiency", "intermediate"), 0.7)
        # log-dampened endorsements/duration so a handful of outlier values
        # (e.g. 56 endorsements) don't dominate; duration capped at 5 years
        depth = (1 + min(endorsements, 60) / 20) * (1 + min(duration, 60) / 24)
        total += proficiency_weight * depth
    # normalize: a candidate with ~6 strong core skills should approach 1.0
    normalized = min(total / 6.0, 1.0) if count else 0.0
    return normalized, count


def _detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    reasons = []
    skills = candidate.get("skills", [])

    expert_low_duration = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 99) <= config.HONEYPOT_EXPERT_MAX_DURATION_MONTHS
    ]
    if expert_low_duration:
        reasons.append(
            f"expert-level proficiency claimed in {len(expert_low_duration)} "
            f"skill(s) with <= {config.HONEYPOT_EXPERT_MAX_DURATION_MONTHS} "
            f"months of use ({', '.join(expert_low_duration[:3])})"
        )

    advanced_low_duration = [
        s["name"] for s in skills
        if s.get("proficiency") == "advanced"
        and s.get("duration_months", 99) <= config.HONEYPOT_ADVANCED_MAX_DURATION_MONTHS
    ]
    if advanced_low_duration:
        reasons.append(
            f"advanced proficiency claimed in {len(advanced_low_duration)} "
            f"skill(s) with <= {config.HONEYPOT_ADVANCED_MAX_DURATION_MONTHS} "
            f"month(s) of use"
        )

    n_expert = sum(1 for s in skills if s.get("proficiency") == "expert")
    if n_expert >= config.HONEYPOT_MIN_EXPERT_SKILL_COUNT:
        reasons.append(
            f"implausibly broad expertise: {n_expert} skills at 'expert' "
            f"proficiency simultaneously"
        )

    # overlapping full-time roles (two roles active at once with no overlap
    # explanation -- a soft signal, only flagged combined with another reason
    # to avoid penalizing legitimate consulting/contract overlap)
    history = sorted(
        [ch for ch in candidate.get("career_history", [])],
        key=lambda x: x["start_date"],
    )
    overlap_months = 0
    for i in range(len(history) - 1):
        end_i = history[i]["end_date"]
        if end_i is None:
            continue
        if history[i + 1]["start_date"] < end_i:
            overlap_months += 1
    if overlap_months and reasons:
        reasons.append(f"{overlap_months} overlapping full-time role period(s)")

    return (len(reasons) > 0), reasons


def _detect_career_traps(candidate: dict) -> tuple[bool, bool, bool, float, int]:
    """Returns (is_consulting_only, is_title_chaser, is_pure_research,
    avg_tenure_months, n_roles)."""
    history = candidate.get("career_history", [])
    n_roles = len(history)
    avg_tenure = (
        sum(ch["duration_months"] for ch in history) / n_roles if n_roles else 0
    )

    companies = {_norm(ch["company"]) for ch in history} | {
        _norm(candidate["profile"]["current_company"])
    }
    is_consulting_only = bool(companies) and companies.issubset(
        config.CONSULTING_FIRMS
    )

    is_title_chaser = (
        n_roles >= config.TITLE_CHASER_MIN_ROLES
        and avg_tenure < config.TITLE_CHASER_AVG_TENURE_MONTHS
    )

    # Pure research: every role's title/industry signals academic/research
    # work with no production deployment language anywhere in the history.
    blob = _career_text_blob(candidate).lower()
    research_titles = sum(
        1 for ch in history
        if re.search(r"\b(research scientist|research intern|phd researcher|"
                      r"postdoc|research fellow)\b", ch["title"], re.IGNORECASE)
    )
    has_production_lang = bool(PRODUCTION_VERBS.search(blob)) or bool(
        re.search(r"\bproduction\b|\breal users\b|\blive\b", blob)
    )
    is_pure_research = (
        n_roles > 0
        and research_titles == n_roles
        and not has_production_lang
    )

    return is_consulting_only, is_title_chaser, is_pure_research, avg_tenure, n_roles


def _education_score(education: list[dict]) -> float:
    if not education:
        return 0.3  # neutral-low; JD doesn't require a degree explicitly
    tier_weight = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45,
                   "unknown": 0.5}
    degree_bonus = {"phd": 0.15, "ph.d": 0.15, "m.s.": 0.08, "m.sc": 0.08,
                     "m.tech": 0.05}
    best = 0.0
    for e in education:
        t = tier_weight.get(e.get("tier", "unknown"), 0.5)
        d = e.get("degree", "").lower()
        bonus = next((v for k, v in degree_bonus.items() if k in d), 0.0)
        best = max(best, min(t + bonus, 1.0))
    return best


def _location_fit(profile: dict, signals: dict) -> float:
    loc = _norm(profile.get("location", ""))
    country = _norm(profile.get("country", ""))
    in_preferred = any(city in loc for city in config.PREFERRED_LOCATIONS_INDIA)

    if country != "india":
        # JD: outside India is case-by-case, no visa sponsorship -- a real
        # but soft negative unless they're willing to relocate.
        base = 0.35 if signals.get("willing_to_relocate") else 0.15
    elif in_preferred and ("pune" in loc or "noida" in loc):
        base = 1.0
    elif in_preferred:
        base = 0.8  # Tier-1 Indian city, JD explicitly welcomes
    else:
        base = 0.55 if signals.get("willing_to_relocate") else 0.35

    notice = signals.get("notice_period_days", 60)
    if notice <= config.NOTICE_PERIOD_IDEAL_DAYS:
        notice_factor = 1.0
    elif notice <= 60:
        notice_factor = 0.85
    else:
        notice_factor = 0.65

    return min(base * (0.7 + 0.3 * notice_factor), 1.0)


def _seniority_authenticity(profile: dict, history: list[dict]) -> tuple[bool, bool]:
    """JD: penalize seniors who've moved into pure architecture/tech-lead
    roles and haven't written code in 18+ months. We approximate this from
    the CURRENT title + description only (we don't have a 'last wrote code'
    field, so title/description language is the best available proxy)."""
    title = _norm(profile.get("current_title", ""))
    is_management_title = any(m in title for m in config.MANAGEMENT_ONLY_TITLE_MARKERS)

    current_desc = ""
    for ch in history:
        if ch.get("is_current"):
            current_desc = ch.get("description", "")
            break
    codes_currently = bool(
        re.search(r"\b(code|coding|implement|wrote|built|engineered|"
                   r"pipeline|model|system)\b", current_desc, re.IGNORECASE)
    ) or not is_management_title

    return is_management_title, codes_currently


def extract_features(candidate: dict) -> CandidateFeatures:
    profile = candidate["profile"]
    skills = candidate.get("skills", [])
    skill_lookup = _normalized_skill_set(skills)
    signals = candidate.get("redrob_signals", {})

    core_score, n_core = _skill_trust_score(skills, config.CORE_SKILLS)
    nice_score, _ = _skill_trust_score(skills, config.NICE_TO_HAVE_SKILLS)
    cv_score, _ = _skill_trust_score(skills, config.CV_SPEECH_SKILLS)

    has_nlp_ir = any(
        name in skill_lookup
        for name in ("nlp", "natural language processing", "information retrieval",
                      "information retrieval systems", "semantic search",
                      "vector search", "embeddings", "bm25", "learning to rank")
    )

    career_score, has_prod, has_eval, has_scale = _compute_career_relevance(candidate)

    is_consulting_only, is_title_chaser, is_pure_research, avg_tenure, n_roles = (
        _detect_career_traps(candidate)
    )

    is_management_only, codes_currently = _seniority_authenticity(
        profile, candidate.get("career_history", [])
    )

    edu_score = _education_score(candidate.get("education", []))
    loc_score = _location_fit(profile, signals)

    is_honeypot, honeypot_reasons = _detect_honeypot(candidate)

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        years_of_experience=profile.get("years_of_experience", 0.0),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        core_skill_trust_score=core_score,
        nice_to_have_trust_score=nice_score,
        cv_speech_trust_score=cv_score,
        has_nlp_or_ir_skill=has_nlp_ir,
        n_core_skills=n_core,
        career_relevance_score=career_score,
        has_shipped_production_signal=has_prod,
        has_eval_framework_signal=has_eval,
        has_scale_signal=has_scale,
        avg_tenure_months=avg_tenure,
        n_roles=n_roles,
        is_title_chaser=is_title_chaser,
        is_consulting_only=is_consulting_only,
        is_pure_research=is_pure_research,
        current_role_is_management_only=is_management_only,
        current_role_codes=codes_currently,
        education_score=edu_score,
        location_fit_score=loc_score,
        notice_period_days=signals.get("notice_period_days", 999),
        open_to_work=signals.get("open_to_work_flag", False),
        last_active_date=signals.get("last_active_date", ""),
        recruiter_response_rate=signals.get("recruiter_response_rate", 0.0),
        github_activity_score=signals.get("github_activity_score", -1.0),
        profile_completeness=signals.get("profile_completeness_score", 0.0),
        interview_completion_rate=signals.get("interview_completion_rate", 0.0),
        verified_email=signals.get("verified_email", False),
        verified_phone=signals.get("verified_phone", False),
        is_honeypot=is_honeypot,
        honeypot_reasons=honeypot_reasons,
        raw=candidate,
    )
