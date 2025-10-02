#!/usr/bin/env python3
"""
Screen subjects for potential exclusion due to strong bias or low sensitivity.

Reads a raw per-trial CSV and computes per-subject hit/false-alarm rates (Jeffreys
smoothed), SDT metrics (d', criterion), and posterior tail probabilities using
Beta-Binomial conjugacy. Outputs a CSV with flags and summary metrics.

Example:
  python scripts/qc_screen_subjects.py \
    --raw_csv downloaded_data/retrocuepilot_sept10.csv \
    --type_col trial_type_label --acc_col is_correct \
    --out_csv qc_subject_bias_screen.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import beta, norm


def _pick_id_columns(df: pd.DataFrame) -> str:
    candidates = [
        c
        for c in df.columns
        if str(c).lower() in (
            "uid",
            "id",
            "prolific_pid",
            "participant_id",
            "subject",
            "subj",
            "worker_id",
        )
    ]
    if not candidates:
        raise ValueError("No id-like column found.")
    pref = [c for c in candidates if str(c).lower() == "uid"]
    return pref[0] if pref else candidates[0]


def _normalize_type_labels(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace(
        {
            "valid": "congruent_true",
            "invalid": "congruent_false",
            "congruenttrue": "congruent_true",
            "congruentfalse": "congruent_false",
            "incong": "incongruent",
            "novel": "incongruent",
            "lure": "incongruent",
        }
    )


def compute_subject_qc(
    raw_csv_path: str,
    out_csv_path: str = "qc_subject_bias_screen.csv",
    id_col: Optional[str] = None,
    type_col: str = "trial_type_label",
    acc_col: str = "is_correct",
    # thresholds
    fa_thresh: float = 0.40,  # liberal FA threshold
    h_thresh: float = 0.60,  # conservative H (hits) threshold
    c_thresh: float = 0.50,  # |criterion| threshold
    dprime_thresh: float = 0.25,  # near-chance threshold
    post_prob: float = 0.95,  # posterior probability cutoff
) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # Fallbacks for common column names if not provided/defaults missing
    if id_col is None:
        try:
            id_col = _pick_id_columns(df)
        except Exception:
            # fallback to first column if nothing recognized
            id_col = df.columns[0]

    if type_col not in df.columns:
        # try common alternatives
        for alt in ("label", "trial_type", "type", "condition"):
            if alt in df.columns:
                type_col = alt
                break
        else:
            raise ValueError(f"Column {type_col!r} not found and no alternative present.")

    if acc_col not in df.columns:
        for alt in ("is_correct", "correct", "acc", "accuracy", "response_correct"):
            if alt in df.columns:
                acc_col = alt
                break
        else:
            raise ValueError(f"Column {acc_col!r} not found and no alternative present.")

    sub = df[[id_col, type_col, acc_col]].copy()
    sub[type_col] = _normalize_type_labels(sub[type_col])
    # coerce accuracy to 0/1, drop missing
    sub[acc_col] = pd.to_numeric(sub[acc_col], errors="coerce")
    sub = sub.dropna(subset=[acc_col])
    sub[acc_col] = sub[acc_col].clip(0, 1).astype(int)

    # keep recognized types
    sub = sub[sub[type_col].isin(["congruent_true", "congruent_false", "incongruent"])].copy()

    # counts per subject × type
    grp = sub.groupby([id_col, type_col])[acc_col].agg(["count", "sum"]).reset_index()
    piv = grp.pivot_table(index=[id_col], columns=type_col, values=["count", "sum"], fill_value=0)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index().rename(columns={id_col: "subject_id"})

    # "Seen" (valid+invalid) totals
    piv["n_old"] = piv.get("count_congruent_true", 0) + piv.get("count_congruent_false", 0)
    piv["y_old"] = piv.get("sum_congruent_true", 0) + piv.get("sum_congruent_false", 0)

    # Novel totals
    piv["n_novel"] = piv.get("count_incongruent", 0)
    piv["y_novel"] = piv.get("sum_incongruent", 0)

    # Jeffreys-smoothed rates
    piv["hit_rate"] = (piv["y_old"] + 0.5) / (piv["n_old"] + 1.0)
    piv["cr_rate"] = (piv["y_novel"] + 0.5) / (piv["n_novel"] + 1.0)
    piv["fa_rate"] = 1.0 - piv["cr_rate"]

    # SDT
    H = piv["hit_rate"].clip(1e-6, 1 - 1e-6)
    F = piv["fa_rate"].clip(1e-6, 1 - 1e-6)
    z = norm.ppf
    piv["d_prime"] = z(H) - z(F)
    piv["criterion_c"] = -0.5 * (z(H) + z(F))

    # Also compute an unsmoothed version of criterion (uses raw rates y/n, clipped only for z)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rate_raw = np.where(piv["n_old"] > 0, piv["y_old"] / piv["n_old"], np.nan)
        cr_rate_raw = np.where(piv["n_novel"] > 0, piv["y_novel"] / piv["n_novel"], np.nan)
    fa_rate_raw = 1.0 - cr_rate_raw
    # clip to avoid infinities when passing into norm.ppf, but do not add Jeffreys prior
    H_raw = np.clip(hit_rate_raw.astype(float), 1e-6, 1 - 1e-6)
    F_raw = np.clip(fa_rate_raw.astype(float), 1e-6, 1 - 1e-6)
    piv["criterion_c_raw"] = -0.5 * (z(H_raw) + z(F_raw))

    # Posterior tails with Jeffreys prior for FA rate
    FA_count = (piv["n_novel"] - piv["y_novel"]).clip(lower=0).astype(float)
    CR_count = piv["y_novel"].astype(float)
    piv[f"post_P_FA_gt_{fa_thresh:.2f}"] = 1 - beta.cdf(fa_thresh, 0.5 + FA_count, 0.5 + CR_count)

    # Posterior tail for hit rate < h_thresh
    hits = piv["y_old"].astype(float)
    misses = (piv["n_old"] - piv["y_old"]).clip(lower=0).astype(float)
    piv[f"post_P_H_lt_{h_thresh:.2f}"] = beta.cdf(h_thresh, 0.5 + hits, 0.5 + misses)

    # Flags
    piv["flag_liberal_FA"] = piv[f"post_P_FA_gt_{fa_thresh:.2f}"] > post_prob
    piv["flag_liberal_c"] = piv["criterion_c"] < -c_thresh
    piv["flag_conserv_c"] = piv["criterion_c"] > c_thresh
    piv["flag_low_hits"] = piv[f"post_P_H_lt_{h_thresh:.2f}"] > post_prob
    # Requested notes/flags
    piv["flag_abs_c_gt_thresh"] = (piv["criterion_c"].abs() > c_thresh)
    piv["flag_dprime_lt_thresh"] = (piv["d_prime"] < dprime_thresh)

    # Combined flags (per request: |c|>c_thresh OR d'<dprime_thresh)
    piv["liberal_flag"] = piv[["flag_liberal_FA", "flag_liberal_c"]].any(axis=1)
    piv["conservative_flag"] = piv[["flag_conserv_c", "flag_low_hits"]].any(axis=1)
    piv["exclude_any"] = piv[["flag_abs_c_gt_thresh", "flag_dprime_lt_thresh"]].any(axis=1)

    # Save
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    piv.to_csv(out_csv_path, index=False)
    return piv


def compute_monkey_qc(
    agg_csv_path: str,
    out_csv_path: str = "qc_monkey_bias_screen.csv",
    c_thresh: float = 0.50,
    dprime_thresh: float = 0.25,
) -> pd.DataFrame:
    """Compute SDT metrics (d', criterion c) for monkeys from aggregated CSV.

    Expects columns per row (can have multiple rows per monkey_id):
      - monkey_id
      - congruent_true_trials, congruent_false_trials, incongruent_trials
      - congruent_true_correct, congruent_false_correct, incongruent_correct
    """
    df = pd.read_csv(agg_csv_path)

    # Sum counts across rows per monkey
    cols_map = {
        "congruent_true_trials": "ct_trials",
        "congruent_false_trials": "cf_trials",
        "incongruent_trials": "inc_trials",
        "congruent_true_correct": "ct_correct",
        "congruent_false_correct": "cf_correct",
        "incongruent_correct": "inc_correct",
    }
    need_cols = set(["monkey_id", *cols_map.keys()])
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Aggregated monkey CSV missing columns: {missing}")

    g = (
        df.rename(columns=cols_map)
          .groupby("monkey_id", as_index=False)[list(cols_map.values())]
          .sum()
    )

    # Old (valid + invalid) vs Novel (incongruent)
    g["n_old"] = g["ct_trials"] + g["cf_trials"]
    g["y_old"] = g["ct_correct"] + g["cf_correct"]
    g["n_novel"] = g["inc_trials"]
    g["y_novel"] = g["inc_correct"]

    # Jeffreys-smoothed rates
    g["hit_rate"] = (g["y_old"] + 0.5) / (g["n_old"] + 1.0)
    g["cr_rate"] = (g["y_novel"] + 0.5) / (g["n_novel"] + 1.0)
    g["fa_rate"] = 1.0 - g["cr_rate"]

    # SDT metrics (clip for stability)
    z = norm.ppf
    H = np.clip(g["hit_rate"].astype(float), 1e-6, 1 - 1e-6)
    F = np.clip(g["fa_rate"].astype(float), 1e-6, 1 - 1e-6)
    g["d_prime"] = z(H) - z(F)
    g["criterion_c"] = -0.5 * (z(H) + z(F))

    out = g[[
        "monkey_id", "n_old", "y_old", "n_novel", "y_novel",
        "hit_rate", "cr_rate", "fa_rate", "d_prime", "criterion_c"
    ]].copy()

    # Flags per request: exclude only on |c| > c_thresh or d' < dprime_thresh
    out["flag_abs_c_gt_thresh"] = (out["criterion_c"].abs() > c_thresh)
    out["flag_dprime_lt_thresh"] = (out["d_prime"] < dprime_thresh)
    out["exclude_any"] = out[["flag_abs_c_gt_thresh", "flag_dprime_lt_thresh"]].any(axis=1)

    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(out_csv_path, index=False)
    return out


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC screen for biased/non-sensitive subjects")
    ap.add_argument("--raw_csv", required=True, help="Path to raw per-trial CSV")
    ap.add_argument("--out_csv", default="qc_subject_bias_screen.csv")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--type_col", default="trial_type_label")
    ap.add_argument("--acc_col", default="is_correct")
    ap.add_argument("--fa_thresh", type=float, default=0.40)
    ap.add_argument("--h_thresh", type=float, default=0.60)
    ap.add_argument("--c_thresh", type=float, default=0.50)
    ap.add_argument("--dprime_thresh", type=float, default=0.25)
    ap.add_argument("--post_prob", type=float, default=0.95)
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    # Auto-detect aggregated monkey format; otherwise process human per-trial
    try:
        head = pd.read_csv(args.raw_csv, nrows=5)
        need_monkey_cols = {
            "monkey_id",
            "congruent_true_trials", "congruent_false_trials", "incongruent_trials",
            "congruent_true_correct", "congruent_false_correct", "incongruent_correct",
        }
        if need_monkey_cols.issubset(set(map(str, head.columns))):
            piv = compute_monkey_qc(
                args.raw_csv,
                out_csv_path=args.out_csv,
                c_thresh=args.c_thresh,
                dprime_thresh=args.dprime_thresh,
            )
            n_excl = int(piv["exclude_any"].sum()) if "exclude_any" in piv.columns else 0
            print(f"Wrote {args.out_csv} — {n_excl}/{len(piv)} monkeys flagged for potential exclusion.")
            return
    except Exception:
        pass

    # Single overall QC screening for humans (per subject across valid/invalid vs novel)
    piv = compute_subject_qc(
        raw_csv_path=args.raw_csv,
        out_csv_path=args.out_csv,
        id_col=args.id_col,
        type_col=args.type_col,
        acc_col=args.acc_col,
        fa_thresh=args.fa_thresh,
        h_thresh=args.h_thresh,
        c_thresh=args.c_thresh,
        dprime_thresh=args.dprime_thresh,
        post_prob=args.post_prob,
    )
    n = len(piv)
    n_excl = int(piv["exclude_any"].sum())
    print(f"Wrote {args.out_csv} — {n_excl}/{n} subjects flagged for potential exclusion.")


if __name__ == "__main__":
    main()


