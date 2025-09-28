#!/usr/bin/env python3
import argparse
import os
import json
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_emulator_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Emulator JSON not found at {path}")
    with open(path, "r") as f:
        emu = json.load(f)
    required = [
        "D",
        "H1",
        "H2",
        "O",
        "x_mean",
        "x_scale",
        "W1",
        "b1",
        "W2",
        "b2",
        "W3",
        "b3",
        "eta_lo",
        "eta_hi",
        "kap_lo",
        "kap_hi",
        "nu_lo",
        "nu_hi",
    ]
    for k in required:
        if k not in emu:
            raise ValueError(f"Missing key in emulator JSON: {k}")

    D = int(emu["D"])
    if D != 8:
        # Expecting feature order: [log10_eta, log10_kappa, log10_nu, q, a, m_2, m_3, m_4]
        raise ValueError(
            f"Emulator D must be 8 with features [log10_eta, log10_kappa, log10_nu, q, a, m_2, m_3, m_4] (got {D})."
        )

    # Soft check on hidden sizes (user noted best is 128x128)
    try:
        if int(emu["H1"]) != 128 or int(emu["H2"]) != 128:
            print(
                f"[warn] emulator hidden sizes are ({emu['H1']},{emu['H2']}); best reported was 128x128."
            )
    except Exception:
        pass

    # optional feature_names validation
    feat_names = emu.get("feature_names", None)
    if feat_names is not None:
        expected = [
            "log10_eta",
            "log10_kappa",
            "log10_nu",
            "q",
            "a",
            "m_2",
            "m_3",
            "m_4",
        ]
        if list(feat_names) != expected:
            print(
                f"[warn] emulator feature_names mismatch. Got {feat_names}, expected {expected}. Proceeding regardless."
            )

    return emu


def _normalize_species(x: str) -> int:
    """Map species -> {1: monkey, 2: human}"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().lower()
    if s.startswith("monkey"):
        return 1
    if s.startswith("human"):
        return 2
    raise ValueError(
        f"Unknown species value: {x!r} (expected 'monkey' or 'human' or 1/2)"
    )


def prepare_stan_long(path_csv: str) -> Tuple[Dict, List[str]]:
    """
    Expect a long-form CSV with at least:
      id, species, q, m, pre_ms, type, n, y
    - type: either {0,1} or strings {congruent_true,congruent_false,valid,invalid}
    """
    df = pd.read_csv(path_csv)

    needed = ["id", "species", "q", "m", "pre_ms", "type", "n", "y"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing column: {col}")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["species_id"] = df["species"].apply(_normalize_species).astype(int)

    # type normalization
    if df["type"].dtype == object:
        tmap = {
            "congruent_true": 1,
            "valid": 1,
            "congruent_false": 0,
            "invalid": 0,
        }
        df["type"] = (
            df["type"].astype(str).str.strip().str.lower().map(tmap)
        )
    df["type"] = df["type"].astype(int)
    if not set(df["type"].unique()).issubset({0, 1}):
        raise ValueError("Column 'type' must be 0/1 or valid/invalid strings.")

    # Loads & timings
    df["m"] = df["m"].astype(float)
    df["a"] = (df["pre_ms"].astype(float) / 1350.0).clip(0, 1)

    # Build participant index (monkeys first, then humans)
    monkeys = sorted(df.loc[df.species_id == 1, "id"].unique().tolist())
    humans = sorted(df.loc[df.species_id == 2, "id"].unique().tolist())
    part_ids = monkeys + humans
    pid_index: Dict[str, int] = {pid: i + 1 for i, pid in enumerate(part_ids)}
    species_id = np.array([1] * len(monkeys) + [2] * len(humans), dtype=int)

    df["pid"] = df["id"].map(pid_index).astype(int)
    df = df.sort_values(["pid", "q", "m", "a", "type"]).reset_index(drop=True)

    K = df.shape[0]
    J = len(part_ids)
    S = 2

    data = dict(
        K=int(K),
        J=int(J),
        S=int(S),
        pid=df["pid"].to_numpy(dtype=int),
        species_id=species_id,
        type=df["type"].to_numpy(dtype=int),
        y=df["y"].to_numpy(dtype=int),
        n=df["n"].to_numpy(dtype=int),
        q=df["q"].to_numpy(dtype=float),
        m=df["m"].to_numpy(dtype=float),
        a=df["a"].to_numpy(dtype=float),
    )
    return data, part_ids


def build_data(args) -> Dict:
    print(f"[fit_with_ma] Loading emulator JSON: {args.emulator_json}", flush=True)
    emu = load_emulator_json(args.emulator_json)
    print(f"[fit_with_ma] Reading and preparing CSV: {args.csv}", flush=True)
    hier, _ = prepare_stan_long(args.csv)

    hier.update(
        dict(
            mu_mean=float(args.mu_mean),
            mu_sd=float(args.mu_sd),
            sigma_rate=float(args.sigma_rate),
            lkj_eta=float(args.lkj_eta),
            beta_sd=float(args.beta_sd),
            D=int(emu["D"]),
            H1=int(emu["H1"]),
            H2=int(emu["H2"]),
            O=int(emu["O"]),
            x_mean=np.array(emu["x_mean"], dtype=float),
            x_scale=np.array(emu["x_scale"], dtype=float),
            W1=np.array(emu["W1"], dtype=float),
            b1=np.array(emu["b1"], dtype=float),
            W2=np.array(emu["W2"], dtype=float),
            b2=np.array(emu["b2"], dtype=float),
            W3=np.array(emu["W3"], dtype=float),
            b3=np.array(emu["b3"], dtype=float),
            eta_lo=float(emu["eta_lo"]),
            eta_hi=float(emu["eta_hi"]),
            kap_lo=float(emu["kap_lo"]),
            kap_hi=float(emu["kap_hi"]),
            nu_lo=float(emu["nu_lo"]),
            nu_hi=float(emu["nu_hi"]),
            tau_obs=np.array([args.tau_monkey, args.tau_human], dtype=float),
        )
    )
    return hier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=None,
        help="Single long-form CSV for both species. If omitted, will use FullRun filtered/unfiltered based on --use_filtered.",
    )
    ap.add_argument(
        "--use_filtered",
        action="store_true",
        help="Use QC-filtered FullRun CSV (data/stan_input_long_FullRun_filtered.csv). If --csv is provided, this flag is ignored.",
    )
    ap.add_argument(
        "--emulator_json",
        default=os.path.join(PROJECT_ROOT, "emulator_artifacts", "train_emu_FullRun", "emulator_for_stan.json"),
        help="Path to emulator JSON (defaults to FullRun emulator).",
    )
    ap.add_argument(
        "--stan_model",
        default=os.path.join(PROJECT_ROOT, "stan", "capacity_control_memory_robust.stan"),
        help="Path to Stan model (defaults to robust variant).",
    )

    # Priors / hyper
    ap.add_argument("--mu_mean", type=float, default=0.0)
    ap.add_argument("--mu_sd", type=float, default=2.0)
    ap.add_argument("--sigma_rate", type=float, default=1.0)
    ap.add_argument("--lkj_eta", type=float, default=2.0)
    ap.add_argument("--beta_sd", type=float, default=1.0)

    # Overdispersion (per species)
    ap.add_argument("--tau_monkey", type=float, default=0.0)
    ap.add_argument("--tau_human", type=float, default=0.0)

    # Sampling
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "stan_fits_cmdstan_ma"))
    args = ap.parse_args()

    # Resolve CSV default if not provided
    if args.csv is None:
        args.csv = (
            os.path.join(PROJECT_ROOT, "data", "stan_input_long_FullRun_filtered.csv")
            if args.use_filtered
            else os.path.join(PROJECT_ROOT, "data", "stan_input_long_FullRun.csv")
        )
        print(f"[fit_with_ma] Auto-selected CSV: {args.csv}")

    print("[fit_with_ma] Building Stan data ...", flush=True)
    data = build_data(args)
    try:
        print(
            f"[fit_with_ma] Data ready: K={data.get('K')}, J={data.get('J')}, D={data.get('D')}, H1={data.get('H1')}, H2={data.get('H2')}",
            flush=True,
        )
    except Exception:
        pass

    # --- CmdStanPy fit ---
    from cmdstanpy import CmdStanModel

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[fit_with_ma] Compiling Stan model (no force rebuild): {args.stan_model}", flush=True)
    model = CmdStanModel(stan_file=args.stan_model)
    model.compile(force=False)
    print("[fit_with_ma] Compile done. Starting sampling ...", flush=True)
    fit = model.sample(
        data=data,
        chains=args.chains,
        parallel_chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.samples,
        seed=args.seed,
        show_progress=True,
        show_console=True,
        output_dir=args.out_dir,
    )

    # Save essentials
    print("[fit_with_ma] Sampling complete. Saving outputs ...", flush=True)
    save = {}
    for name in [
        "mu_monkey",
        "beta_human",
        "sigma_monkey",
        "sigma_human",
        "L_Rho",
        "nu_log",
    ]:
        try:
            save[name] = fit.stan_variable(name)
        except Exception:
            pass

    # Optional robust-model parameters (try/catch so it works for both models)
    for name in [
        "mu_tau_log",      # (draws, 2)
        "sigma_tau_log",   # (draws, 2)
        "epsilon_logit",   # (draws, J)
        "tau_subj",        # (draws, J)
        "epsilon",         # (draws, J)
    ]:
        try:
            save[name] = fit.stan_variable(name)
        except Exception:
            pass

    try:
        log_etakappa = fit.stan_variable("log_etakappa")  # draws x 2 x J
        save["mean_log_etakappa"] = np.mean(log_etakappa, axis=0)  # 2 x J
    except Exception:
        pass

    out_npz = os.path.join(args.out_dir, "draws_basic_ma.npz")
    np.savez_compressed(out_npz, **save)
    print(f"[fit_with_ma] Saved draws to {out_npz}", flush=True)


if __name__ == "__main__":
    main()


