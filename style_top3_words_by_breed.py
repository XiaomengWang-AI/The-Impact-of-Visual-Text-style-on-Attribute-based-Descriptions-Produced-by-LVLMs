"""
Top-K adjectives per breed and per style (no elbow): fixed K=3 by default.

For each breed, independently:
  - Functional style: rank adjectives by P(w) = count_functional / sum_functional (descending).
  - Decorative style: rank by Q(w) = count_decorative / sum_decorative (descending).

Outputs are saved as CSV (long form: every top-3 word per style, not exclusives).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_work_frame(
    df: pd.DataFrame,
    *,
    col_breed: str,
    col_adj: str,
    col_functional: str,
    col_decorative: str,
    min_count_filter: int,
) -> pd.DataFrame:
    """Same normalization as directional_style_topk_analysis._prepare_directional_work_frame."""
    work = df.copy()
    if min_count_filter > 0:
        work = work[
            (work[col_functional] >= min_count_filter)
            | (work[col_decorative] >= min_count_filter)
        ].copy()

    work[col_breed] = work[col_breed].astype(str).str.strip()
    work[col_adj] = work[col_adj].astype(str).str.strip()

    work["functional_total"] = work.groupby(col_breed)[col_functional].transform("sum")
    work["decorative_total"] = work.groupby(col_breed)[col_decorative].transform("sum")
    work["P"] = np.where(
        work["functional_total"] > 0,
        work[col_functional] / work["functional_total"],
        0.0,
    )
    work["Q"] = np.where(
        work["decorative_total"] > 0,
        work[col_decorative] / work["decorative_total"],
        0.0,
    )
    return work


def run_topk_words_per_style(
    df: pd.DataFrame,
    *,
    col_breed: str = "breed",
    col_adj: str = "adjective",
    col_functional: str = "sans_serif",
    col_decorative: str = "cursive",
    min_count_filter: int = 0,
    k: int = 3,
) -> pd.DataFrame:
    """
    Returns long table: breed, style, rank, adjective, prob, count, style_total.
    """
    work = _prepare_work_frame(
        df,
        col_breed=col_breed,
        col_adj=col_adj,
        col_functional=col_functional,
        col_decorative=col_decorative,
        min_count_filter=min_count_filter,
    )

    rows: list[dict] = []
    kk = max(1, int(k))

    for breed, g in work.groupby(col_breed, sort=False):
        g = g.copy()

        by_p = g.sort_values(["P", col_adj], ascending=[False, True]).head(kk)
        for rank, (_, row) in enumerate(by_p.iterrows(), start=1):
            rows.append(
                {
                    "breed": breed,
                    "style": "functional",
                    "rank": rank,
                    "adjective": row[col_adj],
                    "prob": float(row["P"]),
                    "count": float(row[col_functional]),
                    "style_total": float(row["functional_total"]),
                }
            )

        by_q = g.sort_values(["Q", col_adj], ascending=[False, True]).head(kk)
        for rank, (_, row) in enumerate(by_q.iterrows(), start=1):
            rows.append(
                {
                    "breed": breed,
                    "style": "decorative",
                    "rank": rank,
                    "adjective": row[col_adj],
                    "prob": float(row["Q"]),
                    "count": float(row[col_decorative]),
                    "style_total": float(row["decorative_total"]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["breed", "style", "rank"], ascending=[True, True, True]).reset_index(
            drop=True
        )
    return out


def run_top3_wide_by_breed(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per breed: top-3 words per style as separate columns (optional export)."""
    if long_df.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for breed, sub in long_df.groupby("breed", sort=False):
        row: dict[str, str | float] = {"breed": breed}
        for style in ("functional", "decorative"):
            s = sub[sub["style"] == style].sort_values("rank")
            for _, r in s.iterrows():
                rk = int(r["rank"])
                row[f"{style}_rank{rk}_word"] = r["adjective"]
                row[f"{style}_rank{rk}_prob"] = r["prob"]
        parts.append(pd.DataFrame([row]))

    return pd.concat(parts, ignore_index=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(here / "word_freq_input3_qwen.csv"),
        help="Word-count CSV (breed, adjective, sans_serif, cursive, ...). "
        "Functional uses sans_serif counts; decorative uses cursive counts.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(here / "qwen_style_bias_data" / "top3_by_breed"),
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--out-long",
        default="top3_words_long.csv",
        help="Long-format filename under --out-dir.",
    )
    parser.add_argument(
        "--out-wide",
        default="top3_words_wide.csv",
        help="Wide-format filename (one row per breed).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Drop rows only if BOTH counts are below this (default: 0, off).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="How many top words per breed per style (default: 3).",
    )
    parser.add_argument(
        "--no-wide",
        action="store_true",
        help="Do not write the wide CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    long_df = run_topk_words_per_style(
        df,
        min_count_filter=args.min_count,
        k=args.top_k,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / args.out_long
    long_df.to_csv(long_path, index=False)
    print("Wrote:", long_path)

    if not args.no_wide:
        wide_df = run_top3_wide_by_breed(long_df)
        wide_path = out_dir / args.out_wide
        wide_df.to_csv(wide_path, index=False)
        print("Wrote:", wide_path)


if __name__ == "__main__":
    main()
