#!/usr/bin/env python3
# train_emulator_maq.py
# Emulator for (p_valid, p_invalid) with inputs:
#   [log10(eta), log10(kappa), log10(nu), q, a, one-hot(m=2,3,4)]

import os
import json
from glob import glob
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump


EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _round_series(s: pd.Series, digits: int) -> pd.Series:
    return s.astype(float).round(digits)


def load_grid_long(
    grid_dir: str,
    qs: Iterable[float] = (0.70, 0.92),
    expected_m_values: Tuple[int, int, int] = (2, 3, 4),
) -> pd.DataFrame:
    """
    Load concatenated grid chunks and prepare dataframe:
      - filter to selected q values (rounded to 2 decimals)
      - clamp probabilities
      - drop duplicate rows for safety
      - add one-hot columns for m in {2,3,4}
      - basic sanity checks on coverage by (m,a,q)
    """
    paths = sorted(glob(os.path.join(grid_dir, "grid_chunk_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No grid_chunk_*.csv in {grid_dir}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    # filter q
    q_keep = [round(float(q), 2) for q in qs]
    df = df[_round_series(df["q"], 2).isin(q_keep)].copy()

    # clamp and deduplicate
    for c in ("p_valid_mean", "p_invalid_mean"):
        df[c] = df[c].clip(EPS, 1 - EPS)

    df = df.drop_duplicates(subset=["eta", "kappa", "nu", "q", "m", "a"]).reset_index(drop=True)

    # one-hot m (ensure all three columns exist)
    dummies = pd.get_dummies(df["m"].astype(int), prefix="m")
    for col in ("m_2", "m_3", "m_4"):
        if col not in dummies:
            dummies[col] = 0.0
    df = pd.concat([df, dummies[["m_2", "m_3", "m_4"]].astype(float)], axis=1)

    # sanity: per-cell counts and missing cells
    a_vals = sorted(df["a"].astype(float).unique().tolist())
    present_cells = (
        df.assign(q=_round_series(df["q"], 2), a=_round_series(df["a"], 6))
          .groupby(["m", "a", "q"]).size().rename("n").reset_index()
    )
    expected_cells = []
    for m in expected_m_values:
        for a in a_vals:
            for q in q_keep:
                expected_cells.append((m, round(a, 6), q))
    pc_key = set((int(r.m), float(r.a), float(r.q)) for r in present_cells.itertuples(index=False))
    missing = [(m, a, q) for (m, a, q) in expected_cells if (m, a, q) not in pc_key]
    if missing:
        print(f"[warn] Missing {len(missing)} (m,a,q) cells after filtering. Examples: {missing[:5]}")
    # brief coverage print
    print("Cell counts (head):")
    print(present_cells.sort_values(["m", "a", "q"]).head(10).to_string(index=False))

    return df


def build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = np.column_stack(
        [
            np.log10(df["eta"].values),
            np.log10(df["kappa"].values),
            np.log10(df["nu"].values),
            df["q"].values.astype(float),
            df["a"].values.astype(float),
            df["m_2"].values.astype(float),
            df["m_3"].values.astype(float),
            df["m_4"].values.astype(float),
        ]
    )
    Y = np.column_stack(
        [
            logit(df["p_valid_mean"].values),
            logit(df["p_invalid_mean"].values),
        ]
    )
    feat_names = [
        "log10_eta",
        "log10_kappa",
        "log10_nu",
        "q",
        "a",
        "m_2",
        "m_3",
        "m_4",
    ]
    return X, Y, feat_names


def stratify_key(df: pd.DataFrame) -> pd.Series:
    a_code = _round_series(df["a"], 6).astype(str)
    return (
        df["m"].astype(int).astype(str)
        + "_"
        + a_code
        + "_"
        + _round_series(df["q"], 2).astype(str)
    )


def make_group_ids(df: pd.DataFrame) -> pd.Series:
    return (
        _round_series(df["eta"], 8).astype(str)
        + "_"
        + _round_series(df["kappa"], 8).astype(str)
        + "_"
        + _round_series(df["nu"], 8).astype(str)
    )


def export_to_stan_json(pipe, out_json: str, feature_names: List[str]) -> dict:
    scaler = pipe.named_steps["standardscaler"]
    mlp = pipe.named_steps["mlpregressor"]

    if not (len(mlp.coefs_) == 3 and len(mlp.intercepts_) == 3):
        raise AssertionError("Expecting 2 hidden layers + output (3 weight matrices and 3 bias vectors)")

    W1, b1 = mlp.coefs_[0], mlp.intercepts_[0]
    W2, b2 = mlp.coefs_[1], mlp.intercepts_[1]
    W3, b3 = mlp.coefs_[2], mlp.intercepts_[2]
    H1 = int(W1.shape[1])
    H2 = int(W2.shape[1])
    D = int(W1.shape[0])
    O = int(W3.shape[1])

    payload = dict(
        D=D,
        H1=H1,
        H2=H2,
        O=O,
        x_mean=scaler.mean_.tolist(),
        x_scale=scaler.scale_.tolist(),
        W1=W1.tolist(),
        b1=b1.tolist(),
        W2=W2.tolist(),
        b2=b2.tolist(),
        W3=W3.tolist(),
        b3=b3.tolist(),
        feature_names=list(feature_names),
        # domain guards (match generator ranges generously)
        eta_lo=0.01,
        eta_hi=20.0,
        kap_lo=1e-3,
        kap_hi=35.0,
        nu_lo=0.25,
        nu_hi=12.0,
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def save_metrics(out_dir: str, metrics: dict) -> None:
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", path)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main(
    grid_dir: str,
    out_dir: str = os.path.join(PROJECT_ROOT, "emulator_artifacts"),
    qs: Iterable[float] = (0.70, 0.92),
    test_size: float = 0.18,
    random_state: int = 0,
    hidden: Tuple[int, int] = (64, 64),
    alpha: float = 3e-4,
    batch_size: int = 512,
    max_iter: int = 400,
    n_iter_no_change: int = 20,
    split_mode: str = "stratified",  # or "grouped"
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = load_grid_long(grid_dir, qs=qs)
    X, Y, feat_names = build_xy(df)

    # split
    if split_mode == "stratified":
        key = stratify_key(df)
        Xtr, Xte, Ytr, Yte = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=key
        )
    elif split_mode == "grouped":
        groups = make_group_ids(df)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(np.zeros(len(df)), groups=groups))
        Xtr, Xte, Ytr, Yte = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
    else:
        raise ValueError("split_mode must be 'stratified' or 'grouped'")

    # model
    if not (isinstance(hidden, tuple) and len(hidden) == 2):
        raise AssertionError("hidden must be a tuple of length 2 (H1,H2)")

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=alpha,
            learning_rate_init=1e-3,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            verbose=False,
        ),
    )
    model.fit(Xtr, Ytr)

    # metrics
    Ztr_hat = model.predict(Xtr)
    Zte_hat = model.predict(Xte)
    rmse_logit_tr = float(np.sqrt(mean_squared_error(Ytr, Ztr_hat)))
    rmse_logit_te = float(np.sqrt(mean_squared_error(Yte, Zte_hat)))

    Ptr_hat = inv_logit(Ztr_hat)
    Pte_hat = inv_logit(Zte_hat)
    Ptr = inv_logit(Ytr)
    Pte = inv_logit(Yte)
    rmse_prob_tr = float(np.sqrt(mean_squared_error(Ptr, Ptr_hat)))
    rmse_prob_te = float(np.sqrt(mean_squared_error(Pte, Pte_hat)))

    print(f"RMSE (logit): train={rmse_logit_tr:.4f}  test={rmse_logit_te:.4f}")
    print(f"RMSE (prob) : train={rmse_prob_tr:.4f}  test={rmse_prob_te:.4f}")

    # save pipeline
    joblib_path = os.path.join(out_dir, "mlp_emulator.joblib")
    dump(model, joblib_path)
    print(f"Saved pipeline → {joblib_path}")

    # export for Stan
    out_json = os.path.join(out_dir, "emulator_for_stan.json")
    export_to_stan_json(model, out_json, feat_names)
    print(f"Exported weights → {out_json}")

    # per-cell RMSE on test split (only meaningful for stratified)
    try:
        te_df = pd.DataFrame(Xte, columns=feat_names)
        te_df["m"] = np.where(te_df["m_4"] == 1, 4, np.where(te_df["m_3"] == 1, 3, 2))
        te_df["q"] = te_df["q"].round(2)
        te_df["a"] = te_df["a"].round(6)
        Pte_df = pd.DataFrame(Pte_hat, columns=["p_valid_hat", "p_invalid_hat"])
        Pte_true = pd.DataFrame(Pte, columns=["p_valid_true", "p_invalid_true"])
        rep = pd.concat(
            [
                te_df[["m", "a", "q"]].reset_index(drop=True),
                Pte_df.reset_index(drop=True),
                Pte_true.reset_index(drop=True),
            ],
            axis=1,
        )
        cell_rmse = (
            rep.groupby(["m", "a", "q"]).apply(
                lambda g: pd.Series(
                    {
                        "rmse_valid": float(np.sqrt(np.mean((g["p_valid_hat"] - g["p_valid_true"]) ** 2))),
                        "rmse_invalid": float(np.sqrt(np.mean((g["p_invalid_hat"] - g["p_invalid_true"]) ** 2))),
                        "n": int(len(g)),
                    }
                )
            )
        ).reset_index()
        cell_rmse = cell_rmse.sort_values(["m", "a", "q"]) 
        cell_rmse_path = os.path.join(out_dir, "cell_rmse.csv")
        cell_rmse.to_csv(cell_rmse_path, index=False)
        print("Wrote per-cell RMSE →", cell_rmse_path)
    except Exception as e:
        print("[warn] per-cell RMSE report skipped:", str(e))

    metrics = {
        "rmse_logit": {"train": rmse_logit_tr, "test": rmse_logit_te},
        "rmse_prob": {"train": rmse_prob_tr, "test": rmse_prob_te},
        "hidden": list(hidden),
        "alpha": float(alpha),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "n_iter_no_change": int(n_iter_no_change),
        "split_mode": split_mode,
        "n_train": int(len(X) - len(Xte)),
        "n_test": int(len(Xte)),
        "features": feat_names,
        "qs": [float(q) for q in qs],
    }
    save_metrics(out_dir, metrics)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train emulator over (m, a, q) conditions")
    ap.add_argument("--grid_dir", required=True, help="Directory containing grid_chunk_*.csv")
    ap.add_argument("--out_dir", default="emulator_artifacts")
    ap.add_argument(
        "--qs",
        default="0.70,0.92",
        help="Comma-separated q values to include (e.g., '0.70,0.92' or '0.92')",
    )
    ap.add_argument("--test_size", type=float, default=0.18)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument(
        "--hidden",
        default="64,64",
        help="Comma-separated sizes for two hidden layers (e.g., '64,64')",
    )
    ap.add_argument("--alpha", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--n_iter_no_change", type=int, default=20)
    ap.add_argument(
        "--split_mode",
        choices=["stratified", "grouped"],
        default="stratified",
        help="'stratified' by (m,a,q) or 'grouped' by (eta,kappa,nu)",
    )
    args = ap.parse_args()

    hidden_tuple = tuple(int(x) for x in args.hidden.split(","))
    if len(hidden_tuple) != 2:
        raise SystemExit("--hidden must specify exactly two integers, e.g. '64,64'")

    qs = [float(x) for x in args.qs.split(",") if x.strip()]

    main(
        grid_dir=args.grid_dir,
        out_dir=args.out_dir,
        qs=qs,
        test_size=args.test_size,
        random_state=args.random_state,
        hidden=hidden_tuple,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        n_iter_no_change=args.n_iter_no_change,
        split_mode=args.split_mode,
    )


