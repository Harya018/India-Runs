"""
Streamlit sandbox app -- satisfies the hackathon's Section 10.5 sandbox
requirement: a hosted environment that runs the ranker end-to-end on a
small candidate sample (<=100 candidates) and produces a ranked CSV.

Run locally:
    streamlit run sandbox/app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub.
    2. On share.streamlit.io, point a new app at this file
       (sandbox/app.py) with the repo's requirements.txt.
    3. The app ships its own small sample (data/sample_candidates.json,
       50 candidates) so it works with zero configuration -- no need to
       upload the full 100k-candidate pool to a free-tier host.
"""

import sys
import json
import io
import csv
import random
from pathlib import Path
from datetime import date

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import extract_features
from src.scoring import score_candidate
from src.reasoning import generate_reasoning

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Candidate Ranker — Sandbox")
st.caption(
    "Runs the same scoring pipeline as rank.py on a small candidate sample. "
    "No network calls, no GPU, pure-Python scoring — see src/ for the full logic."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload a candidates JSON/JSONL sample (<=100 candidates)",
        type=["json", "jsonl"],
    )
    use_bundled = st.checkbox(
        "Use bundled sample (50 candidates)", value=uploaded is None
    )
    reference_date_str = st.text_input("Reference date (YYYY-MM-DD)", value="2026-06-24")

def load_candidates():
    if uploaded is not None and not use_bundled:
        content = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    with open(SAMPLE_PATH) as f:
        return json.load(f)


if st.sidebar.button("Run ranker", type="primary"):
    candidates = load_candidates()
    st.write(f"Loaded **{len(candidates)}** candidates.")

    try:
        y, m, d = (int(x) for x in reference_date_str.split("-"))
        reference_date = date(y, m, d)
    except ValueError:
        st.error("Reference date must be YYYY-MM-DD. Using today instead.")
        reference_date = date.today()

    scored = []
    progress = st.progress(0)
    for i, c in enumerate(candidates):
        feat = extract_features(c)
        sc = score_candidate(feat, reference_date)
        scored.append(sc)
        progress.progress((i + 1) / len(candidates))

    scored.sort(key=lambda sc: (-round(sc.score, 4), sc.candidate_id))
    top = scored[: min(100, len(scored))]

    rng = random.Random(42)
    rows = []
    for rank, sc in enumerate(top, start=1):
        reasoning = generate_reasoning(sc, rank, rng)
        rows.append({
            "rank": rank,
            "candidate_id": sc.candidate_id,
            "score": round(sc.score, 4),
            "title": sc.features.current_title,
            "company": sc.features.current_company,
            "years_exp": sc.features.years_of_experience,
            "reasoning": reasoning,
        })

    st.subheader(f"Ranked top {len(rows)}")
    st.dataframe(rows, use_container_width=True, height=500)

    # Downloadable CSV in the exact submission format
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
    st.download_button(
        "Download ranked CSV (submission format)",
        data=buf.getvalue(),
        file_name="sandbox_submission.csv",
        mime="text/csv",
    )

    with st.expander("Score breakdown for top candidate"):
        top1 = top[0]
        st.json({
            "candidate_id": top1.candidate_id,
            "final_score": top1.score,
            "base_fit_score": top1.base_fit_score,
            "behavioral_multiplier": top1.behavioral_multiplier,
            "trap_penalty_multiplier": top1.trap_penalty_multiplier,
            "core_skill_trust_score": top1.features.core_skill_trust_score,
            "career_relevance_score": top1.features.career_relevance_score,
            "is_honeypot": top1.features.is_honeypot,
            "is_consulting_only": top1.features.is_consulting_only,
            "is_title_chaser": top1.features.is_title_chaser,
        })
else:
    st.info("Click **Run ranker** in the sidebar to score the sample candidates.")
