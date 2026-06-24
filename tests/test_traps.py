"""
Unit tests for the trap-detection and feature-extraction logic.

Run with: python -m pytest tests/ -v
(or: python -m unittest discover tests/  if pytest isn't installed)

These use small synthetic candidate fixtures rather than the real dataset,
so they run instantly and pin down exact expected behavior for each named
trap in job_description.md and the honeypot warning in submission_spec.md.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import extract_features
from src.scoring import score_candidate, base_fit_score
from datetime import date


def _base_candidate(**overrides):
    """Minimal valid candidate record; overrides patch specific fields."""
    c = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Test headline",
            "summary": "Test summary with no special content.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "Internet",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "ML Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "Internet",
                "company_size": "201-500",
                "description": "Built and shipped a production ranking system serving real users.",
            }
        ],
        "education": [
            {
                "institution": "Test University",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2015,
                "end_year": 2019,
                "grade": "8.0 CGPA",
                "tier": "tier_2",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 80.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 30.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for key, val in overrides.items():
        if key in c:
            if isinstance(val, dict) and isinstance(c[key], dict):
                c[key].update(val)
            else:
                c[key] = val
    return c


class TestHoneypotDetection(unittest.TestCase):
    def test_zero_duration_expert_skill_is_honeypot(self):
        c = _base_candidate()
        c["skills"] = [
            {"name": "MLflow", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "Photoshop", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        ]
        feat = extract_features(c)
        self.assertTrue(feat.is_honeypot)

    def test_legitimate_expert_skill_with_real_tenure_not_honeypot(self):
        c = _base_candidate()
        c["skills"] = [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
        ]
        feat = extract_features(c)
        self.assertFalse(feat.is_honeypot)

    def test_many_simultaneous_expert_skills_is_honeypot(self):
        c = _base_candidate()
        c["skills"] = [
            {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 10, "duration_months": 24}
            for i in range(9)
        ]
        feat = extract_features(c)
        self.assertTrue(feat.is_honeypot)


class TestConsultingOnlyTrap(unittest.TestCase):
    def test_entire_career_at_consulting_firms_flagged(self):
        c = _base_candidate()
        c["profile"]["current_company"] = "TCS"
        c["career_history"] = [
            {**c["career_history"][0], "company": "TCS"},
            {**c["career_history"][0], "company": "Infosys", "is_current": False,
             "end_date": "2023-01-01"},
        ]
        feat = extract_features(c)
        self.assertTrue(feat.is_consulting_only)

    def test_current_consulting_but_prior_product_company_not_flagged(self):
        """JD explicitly: 'currently at one of these companies but have
        prior product-company experience, that's fine.'"""
        c = _base_candidate()
        c["profile"]["current_company"] = "TCS"
        c["career_history"] = [
            {**c["career_history"][0], "company": "TCS"},
            {**c["career_history"][0], "company": "Google", "is_current": False,
             "end_date": "2023-01-01"},
        ]
        feat = extract_features(c)
        self.assertFalse(feat.is_consulting_only)


class TestTitleChaserTrap(unittest.TestCase):
    def test_short_average_tenure_with_many_roles_flagged(self):
        c = _base_candidate()
        c["career_history"] = [
            {**c["career_history"][0], "duration_months": 10, "is_current": False,
             "end_date": "2024-01-01"},
            {**c["career_history"][0], "duration_months": 12, "is_current": False,
             "end_date": "2023-01-01"},
            {**c["career_history"][0], "duration_months": 8, "is_current": True,
             "end_date": None},
        ]
        feat = extract_features(c)
        self.assertTrue(feat.is_title_chaser)

    def test_stable_long_tenure_not_flagged(self):
        c = _base_candidate()
        feat = extract_features(c)
        self.assertFalse(feat.is_title_chaser)


class TestSkillCareerCorroboration(unittest.TestCase):
    def test_ai_skills_with_unrelated_career_history_penalized(self):
        """The JD's central example: AI keywords in skills, but career
        history shows an entirely unrelated job function."""
        c = _base_candidate()
        c["profile"]["current_title"] = "Marketing Manager"
        c["career_history"] = [{
            **c["career_history"][0],
            "title": "Marketing Manager",
            "description": "Led brand campaigns and managed a content calendar for social media.",
        }]
        c["skills"] = [
            {"name": "RAG", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 4, "duration_months": 10},
            {"name": "Vector Search", "proficiency": "intermediate", "endorsements": 3, "duration_months": 8},
        ]
        feat = extract_features(c)
        no_history_score = base_fit_score(feat)

        # Compare against an otherwise-identical candidate whose career
        # history DOES corroborate the same skills.
        c2 = _base_candidate()
        c2["skills"] = c["skills"]
        feat2 = extract_features(c2)
        with_history_score = base_fit_score(feat2)

        self.assertLess(no_history_score, with_history_score)

    def test_corroborated_skills_score_highly(self):
        c = _base_candidate()
        c["career_history"][0]["description"] = (
            "Shipped a production hybrid retrieval system combining BM25 "
            "and dense vector recall, serving 10M+ queries/month with an "
            "NDCG@10 evaluation harness."
        )
        c["skills"] = [
            {"name": "BM25", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 8, "duration_months": 20},
        ]
        feat = extract_features(c)
        self.assertGreater(feat.career_relevance_score, 0.5)


class TestPureResearchTrap(unittest.TestCase):
    def test_research_only_titles_with_no_production_language_flagged(self):
        c = _base_candidate()
        c["career_history"] = [{
            **c["career_history"][0],
            "title": "Research Scientist",
            "description": "Published papers on transformer architectures at a university lab.",
        }]
        feat = extract_features(c)
        self.assertTrue(feat.is_pure_research)

    def test_research_title_with_production_deployment_not_flagged(self):
        c = _base_candidate()
        c["career_history"] = [{
            **c["career_history"][0],
            "title": "Research Scientist",
            "description": "Deployed the model to production serving real users at scale.",
        }]
        feat = extract_features(c)
        self.assertFalse(feat.is_pure_research)


class TestScoringMonotonicity(unittest.TestCase):
    def test_inactive_unresponsive_candidate_scores_lower_than_active_twin(self):
        active = _base_candidate()
        inactive = _base_candidate(redrob_signals={
            "last_active_date": "2025-01-01",
            "recruiter_response_rate": 0.05,
            "open_to_work_flag": False,
        })
        feat_active = extract_features(active)
        feat_inactive = extract_features(inactive)
        sc_active = score_candidate(feat_active, date(2026, 6, 24))
        sc_inactive = score_candidate(feat_inactive, date(2026, 6, 24))
        self.assertLess(sc_inactive.score, sc_active.score)
        # base fit (pre-behavioral-multiplier) should be IDENTICAL --
        # behavioral signals must only affect the multiplier layer.
        self.assertAlmostEqual(sc_active.base_fit_score, sc_inactive.base_fit_score)


if __name__ == "__main__":
    unittest.main()
