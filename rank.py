#!/usr/bin/env python3
"""
rank.py -- produce the top-100 ranked submission CSV for the Redrob hackathon.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

Reads the full candidate pool (handles both plain .jsonl and gzipped
.jsonl.gz transparently), scores every candidate with the hybrid scorer in
src/scoring.py, and writes the top 100 as a validator-compliant CSV.

Designed to respect the hackathon's compute constraints: CPU-only, no
network calls, single-pass streaming JSON parse (doesn't load translated
feature objects for all 100k candidates into a heavy DataFrame -- keeps
peak memory low), and completes well under 5 minutes on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import extract_features
from src.scoring import score_candidate
from src.reasoning import generate_reasoning

TOP_N = 100
REASONING_RNG_SEED = 42  # fixed seed -> reproducible phrasing choices, not reproducible content


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: Path):
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def rank_candidates(candidates_path: Path, reference_date: date, verbose: bool = True):
    scored = []
    n = 0
    t0 = time.time()
    for candidate in iter_candidates(candidates_path):
        features = extract_features(candidate)
        sc = score_candidate(features, reference_date)
        scored.append(sc)
        n += 1
        if verbose and n % 20000 == 0:
            print(f"  scored {n:,} candidates ({time.time()-t0:.1f}s elapsed)", file=sys.stderr)

    if verbose:
        print(f"Scored {n:,} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    # Sort descending by score; tie-break by candidate_id ascending per spec.
    # IMPORTANT: tie-break correctness must be evaluated on the score as it
    # will be WRITTEN (rounded to 4 decimals), not the raw float -- two
    # candidates with slightly different raw scores can round to an
    # identical written value, and the validator checks ties against what's
    # actually in the file.
    scored.sort(key=lambda sc: (-round(sc.score, 4), sc.candidate_id))
    return scored[:TOP_N]


def write_submission_csv(top: list, out_path: Path):
    rng = random.Random(REASONING_RNG_SEED)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Round once, up front, and re-clamp to be non-increasing on the
    # ROUNDED values (the values actually written) -- this is what the
    # validator checks, so it's what must be internally consistent.
    rounded_scores = []
    prev = None
    for sc in top:
        r = round(sc.score, 4)
        if prev is not None and r > prev:
            r = prev
        rounded_scores.append(r)
        prev = r

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (sc, score) in enumerate(zip(top, rounded_scores), start=1):
            reasoning = generate_reasoning(sc, rank, rng)
            writer.writerow([sc.candidate_id, rank, f"{score:.4f}", reasoning])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path,
                         help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, type=Path,
                         help="Path to write the ranked submission CSV")
    parser.add_argument("--reference-date", default=None,
                         help="YYYY-MM-DD date to use for recency scoring "
                              "(defaults to today; use a fixed date for "
                              "fully reproducible scores across runs)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.reference_date:
        y, m, d = (int(x) for x in args.reference_date.split("-"))
        reference_date = date(y, m, d)
    else:
        reference_date = date.today()

    t0 = time.time()
    top = rank_candidates(args.candidates, reference_date, verbose=not args.quiet)
    write_submission_csv(top, args.out)
    elapsed = time.time() - t0

    if not args.quiet:
        print(f"\nWrote top {len(top)} candidates to {args.out} in {elapsed:.1f}s total", file=sys.stderr)
        print(f"Top 5 candidate_ids: {[sc.candidate_id for sc in top[:5]]}", file=sys.stderr)


if __name__ == "__main__":
    main()
