"""Build the shared 300-query PinPoint subset (Plan A: every model runs on the same file).

Usage: python scripts/make_pinpoint_subset.py <path/to/pinpoint_licensed.parquet> [out.json]

Sampling is stratified by query_category (proportional, seed 42), so the subset
keeps the benchmark's category mix, including Negation queries.
"""
import json
import sys

import pandas as pd

N, SEED = 300, 42


def url(sig):
    return f"https://i.pinimg.com/736x/{sig[:2]}/{sig[2:4]}/{sig[4:6]}/{sig}.jpg"


def clean(sigs):
    return [s for s in sigs if s and s != "None"]


def main(parquet, out="pinpoint_subset_300.json"):
    df = pd.read_parquet(parquet)
    # Largest-remainder rounding so per-category quotas sum to exactly N.
    share = df.query_category.value_counts() / len(df) * N
    quota = share.astype(int)
    quota[(share - quota).sort_values(ascending=False).index[: N - quota.sum()]] += 1
    sub = pd.concat(
        df[df.query_category == c].sample(int(k), random_state=SEED) for c, k in quota.items()
    ).sort_values("query_id")

    data = {}
    for r in sub.itertuples():
        refs = clean([r.query_image_signature, r.query_image_signature2])
        data[r.query_id.removeprefix("query_").zfill(5)] = {
            "instruction": r.instruction,
            "query_category": r.query_category,
            "ref_image_signatures": refs,
            "ref_image_urls": [url(s) for s in refs],
            "positive_candidates": clean(r.positive_candidates),
            "negative_candidates": clean(r.negative_candidates),
            "is_single": bool(r.is_single),
            "l1_interest": r.l1_interest,
        }

    assert len(data) == N and all(v["ref_image_signatures"] and v["positive_candidates"] for v in data.values())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} queries to {out}\n{quota.to_string()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
