"""
EDA script: the exploration that produced the constants in src/config.py.

Run with: python notebooks/eda.py

This is not meant to be exhaustive -- it's the record of the specific
questions we asked the dataset to ground scoring decisions in actual data
patterns rather than assumptions. Each section below maps directly to a
decision documented in src/config.py or src/scoring.py.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "candidates.jsonl.gz"


def load_candidates():
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    candidates = list(load_candidates())
    print(f"Loaded {len(candidates):,} candidates")

    # -----------------------------------------------------------------
    section("1. Title distribution -- is this really a needle-in-haystack problem?")
    # -----------------------------------------------------------------
    titles = Counter(c["profile"]["current_title"] for c in candidates)
    print("Top 15 titles overall:")
    for t, n in titles.most_common(15):
        print(f"  {n:5d}  {t}")

    ai_ml_titles = Counter(
        c["profile"]["current_title"] for c in candidates
        if any(k in c["profile"]["current_title"]
               for k in ("AI", "ML", "Machine Learning", "Data Scientist", "NLP"))
    )
    print(f"\nAI/ML-titled candidates: {sum(ai_ml_titles.values())} / {len(candidates)} "
          f"({sum(ai_ml_titles.values())/len(candidates):.1%})")
    for t, n in ai_ml_titles.most_common():
        print(f"  {n:5d}  {t}")
    # Finding: ~1% of the pool has an AI/ML-flavored title. The "ideal
    # candidate" the JD describes (Senior/Staff/Lead seniority within
    # this slice) is a tiny fraction of that. Confirms the JD's own
    # statement: "we're not expecting to find many matches in a 100K pool."

    # -----------------------------------------------------------------
    section("2. The Tier-5 trap: plain-language titles with real IR substance")
    # -----------------------------------------------------------------
    keywords = ["ranking", "recommendation system", "retrieval", "embedding",
                "click-through", "NDCG"]
    hits = []
    for c in candidates:
        title = c["profile"]["current_title"]
        if any(k in title for k in ("AI", "ML", "Machine Learning", "Data Scientist", "NLP")):
            continue
        blob = " ".join(ch["description"] for ch in c["career_history"]).lower()
        matched = [k for k in keywords if k.lower() in blob]
        if len(matched) >= 2:
            hits.append((c["candidate_id"], title))
    print(f"Non-AI-titled candidates with >=2 IR domain terms in career history: {len(hits)}")
    title_counts = Counter(t for _, t in hits)
    for t, n in title_counts.most_common():
        print(f"  {n:5d}  {t}")
    # Finding: "Search Engineer" and "Recommendation Systems Engineer" are
    # real titles in this dataset that carry zero AI/ML buzzword but
    # substantial IR substance in career_history. This is the exact
    # "Tier 5" case job_description.md calls out by name. Any
    # keyword-on-title ranker would systematically miss this cohort, which
    # is why career_relevance_score in features.py reads career_history
    # text, not the title field.

    # -----------------------------------------------------------------
    section("3. Honeypot signature: 'expert' proficiency at ~0 duration")
    # -----------------------------------------------------------------
    honeypot_like = []
    for c in candidates:
        zero_dur_experts = [
            s["name"] for s in c.get("skills", [])
            if s["proficiency"] == "expert" and s.get("duration_months", 99) <= 3
        ]
        if zero_dur_experts:
            honeypot_like.append((c["candidate_id"], c["profile"]["current_title"], zero_dur_experts))
    print(f"Candidates with 'expert' skill claimed at <=3 months use: {len(honeypot_like)}")
    for cid, title, skills in honeypot_like[:15]:
        print(f"  {cid}  {title}  {skills}")
    off_domain = sum(
        1 for _, title, _ in honeypot_like
        if not any(k in title for k in ("AI", "ML", "Data Scientist", "NLP"))
    )
    print(f"  -> {off_domain}/{len(honeypot_like)} of these are off-domain titles "
          f"(HR/Accounting/Sales/etc.), consistent with this being a planted "
          f"honeypot signature rather than a real fast-learner pattern.")

    # -----------------------------------------------------------------
    section("4. Title-chaser pattern: average tenure across AI-titled candidates")
    # -----------------------------------------------------------------
    chasers = []
    for c in candidates:
        title = c["profile"]["current_title"]
        if not any(k in title for k in ("AI", "ML", "Machine Learning", "Data Scientist", "NLP")):
            continue
        history = c["career_history"]
        if len(history) >= 3:
            avg_tenure = sum(h["duration_months"] for h in history) / len(history)
            if avg_tenure < 18:
                chasers.append((c["candidate_id"], title, avg_tenure))
    print(f"AI-titled candidates with avg tenure <18mo across >=3 roles: {len(chasers)}")
    for cid, title, avg in chasers[:10]:
        print(f"  {cid}  {title}  avg_tenure={avg:.1f}mo")
    # Finding: ~30 AI-titled candidates show the job_description.md
    # "Senior -> Staff -> Principal" title-chasing pattern via tenure data.
    # We don't have explicit title-progression history, so average tenure
    # is the best available proxy -- documented as such in config.py.

    # -----------------------------------------------------------------
    section("5. Consulting-only trap among AI-titled candidates")
    # -----------------------------------------------------------------
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant",
                         "capgemini", "hcl"}
    consulting_only = []
    for c in candidates:
        title = c["profile"]["current_title"]
        if not any(k in title for k in ("AI", "ML", "Machine Learning", "Data Scientist", "NLP")):
            continue
        companies = {ch["company"].lower() for ch in c["career_history"]} | {
            c["profile"]["current_company"].lower()
        }
        if companies and companies.issubset(consulting_firms):
            consulting_only.append((c["candidate_id"], title, companies))
    print(f"AI-titled candidates whose ENTIRE career is at consulting firms: {len(consulting_only)}")
    for cid, title, companies in consulting_only[:10]:
        print(f"  {cid}  {title}  {companies}")

    # -----------------------------------------------------------------
    section("6. Skill vocabulary frequency bands -- the basis for CORE_SKILLS")
    # -----------------------------------------------------------------
    skill_freq = Counter()
    for c in candidates:
        for s in c.get("skills", []):
            skill_freq[s["name"]] += 1
    bands = {"high (>8000, generic)": 0, "mid (1000-6000, general ML/AI)": 0,
             "low (<10, rare synonyms)": 0}
    for skill, freq in skill_freq.items():
        if freq > 8000:
            bands["high (>8000, generic)"] += 1
        elif 1000 <= freq <= 6000:
            bands["mid (1000-6000, general ML/AI)"] += 1
        elif freq < 10:
            bands["low (<10, rare synonyms)"] += 1
    print("Skill vocabulary splits into clean frequency bands:")
    for band, count in bands.items():
        print(f"  {count:3d} skills in band: {band}")
    rare = sorted((s for s, f in skill_freq.items() if f < 10), key=lambda s: skill_freq[s])
    print(f"\nRare (<10 freq) skills -- these read as deliberate plain-language "
          f"synonyms for buzzword skills, folded into CORE_SKILLS in config.py:")
    for s in rare:
        print(f"  {skill_freq[s]:2d}  {s}")


if __name__ == "__main__":
    main()
