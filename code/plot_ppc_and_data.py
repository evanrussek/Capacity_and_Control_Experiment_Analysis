
#%%
# Setup and imports
import os
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import load

# Global typography (match paper figures)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'STIXGeneral', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11


#%%
# Configuration (edit here when running interactively)
# Resolve project root from this script location; in notebooks __file__ is undefined
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
NPZ_FILE = os.path.join(PROJECT_ROOT, "stan_fits_cmdstan_ma_robust_excl_FullRun", "draws_basic_ma.npz")
EMULATOR_PIPELINE = os.path.join(PROJECT_ROOT, "emulator_artifacts", "train_emu_FullRun", "mlp_emulator.joblib")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_plots")
PREFIX = "ppc_FullRun_filtered"

N_DRAWS = 200
N_SUBJ_HUMAN = 346
N_SUBJ_MONKEY = 10
SEED = 123
T_TOTAL_MS = 1350.0

# Data sources and condition grids
USE_FILTERED_DATA = True
HUMAN_DATA_CSV = None  # if None, will fall back based on USE_FILTERED_DATA
MONKEY_DATA_CSV = None  # if None, will fall back based on USE_FILTERED_DATA
MONKEY_CONDITIONS_CSV = os.path.join(PROJECT_ROOT, "data", "monkey_aggregated_2.csv")

PRE_MS_LIST = [300, 700, 1100]
Q_HUMAN_LIST = [0.70, 0.92]
Q_MONKEY = 0.92

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")

# Data panel uses subject-mean accuracy and CI computed from raw data

#%%
# Helper functions
def flatten_draws(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 1:
        return arr
    if arr.ndim >= 3 and arr.shape[-1] == arr.shape[-2]:
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return arr.reshape(-1, arr.shape[-1])


def build_emulator_predict_fn(pipeline_path: str):
    pipe = load(pipeline_path)
    def predict(log10_eta: float, log10_kap: float, log10_nu: float, q: float, a: float, m: int) -> Tuple[float, float]:
        m2 = 1.0 if m == 2 else 0.0
        m3 = 1.0 if m == 3 else 0.0
        m4 = 1.0 if m == 4 else 0.0
        x = np.array([[log10_eta, log10_kap, log10_nu, q, a, m2, m3, m4]], dtype=float)
        z = pipe.predict(x)[0]
        # pipe is trained in logit space; convert back to probabilities
        p_valid = float(1.0 / (1.0 + np.exp(-z[0])))
        p_invalid = float(1.0 / (1.0 + np.exp(-z[1])))
        return p_valid, p_invalid
    return predict


def sample_subjects(mu: np.ndarray, sigma: np.ndarray, L_Rho: np.ndarray, n_subj: int, rng: np.random.Generator) -> np.ndarray:
    L = np.diag(sigma) @ L_Rho
    z = rng.standard_normal(size=(2, n_subj))
    log_etakappa = mu[:, None] + L @ z  # (2, n_subj)
    return log_etakappa


def compute_cell_means_for_draw(
    species: str,
    mu_draw: np.ndarray,
    sigma_draw: np.ndarray,
    L_Rho_draw: np.ndarray,
    nu_draw: float,
    q_vals: np.ndarray,
    a_vals: np.ndarray,
    m_vals: np.ndarray,
    n_subj: int,
    predict,
    rng: np.random.Generator,
) -> Dict:
    res = {}
    log10 = np.log10
    log_ek = sample_subjects(mu_draw, sigma_draw, L_Rho_draw, n_subj, rng)  # (2, n_subj)
    log10_nu = float(log10(max(1e-8, nu_draw)))
    for ai, a in enumerate(a_vals):
        sub = {"valid": [], "invalid": []}
        for m in m_vals:
            acc_valid = []
            acc_invalid = []
            for j in range(n_subj):
                eta = float(np.exp(log_ek[0, j])); kap = float(np.exp(log_ek[1, j]))
                pv_list = []
                pi_list = []
                for q in q_vals:
                    pv, pi = predict(np.log10(eta), np.log10(kap), log10_nu, float(q), float(a), int(m))
                    pv_list.append(pv); pi_list.append(pi)
                acc_valid.append(float(np.mean(pv_list)))
                acc_invalid.append(float(np.mean(pi_list)))
            sub["valid"].append(float(np.mean(acc_valid)))
            sub["invalid"].append(float(np.mean(acc_invalid)))
        res[ai] = {"m": m_vals.astype(float), "valid": np.array(sub["valid"], float), "invalid": np.array(sub["invalid"], float)}
    return res


def bands_over_draws(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.mean(X, axis=0), np.percentile(X, 2.5, axis=0), np.percentile(X, 97.5, axis=0)


def _data_mean_ci(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    work = df[df["label"].isin(["congruent_true", "congruent_false"])].copy()
    agg = work.groupby(["id", *by_cols])[['y', 'n']].sum().reset_index()
    agg['acc'] = agg['y'].astype(float) / agg['n'].clip(lower=1).astype(float)
    stats = (
        agg.groupby(by_cols)["acc"].agg(mean_acc='mean', std_acc='std', N='count').reset_index()
    )
    stats['sem'] = stats['std_acc'] / np.sqrt(np.maximum(stats['N'], 1))
    stats['ci95'] = 1.96 * stats['sem']
    return stats


def data_human_acc_vs_load(hum_df: pd.DataFrame) -> pd.DataFrame:
    return _data_mean_ci(hum_df, ["m", "label"]).sort_values(["m", "label"]).reset_index(drop=True)


def data_human_acc_vs_prems(hum_df: pd.DataFrame) -> pd.DataFrame:
    return _data_mean_ci(hum_df, ["pre_ms", "label"]).sort_values(["pre_ms", "label"]).reset_index(drop=True)


def data_human_acc_vs_q(hum_df: pd.DataFrame) -> pd.DataFrame:
    return _data_mean_ci(hum_df, ["q", "label"]).sort_values(["q", "label"]).reset_index(drop=True)


#%%
# Load emulator and posterior draws
if not os.path.exists(NPZ_FILE):
    raise FileNotFoundError(f"Could not find draws file: {NPZ_FILE}")
if not os.path.exists(EMULATOR_PIPELINE):
    raise FileNotFoundError(f"Emulator pipeline not found: {EMULATOR_PIPELINE}")

rng = np.random.default_rng(SEED)
d = np.load(NPZ_FILE)
predict = build_emulator_predict_fn(EMULATOR_PIPELINE)

# Extract draws
mu_monkey = flatten_draws(d["mu_monkey"])  # (N,2)
beta_human = flatten_draws(d["beta_human"])  # (N,2)
sigma_monkey = flatten_draws(d["sigma_monkey"])  # (N,2)
sigma_human = flatten_draws(d["sigma_human"])    # (N,2)
if "L_Rho" in d.files:
    L_Rho = flatten_draws(d["L_Rho"])  # (N,2,2)
else:
    L_Rho = np.tile(np.eye(2, dtype=float), (mu_monkey.shape[0], 1, 1))
if "nu" in d.files:
    nu = flatten_draws(d["nu"]).reshape(-1)
elif "nu_log" in d.files:
    nu = np.exp(flatten_draws(d["nu_log"]).reshape(-1))
else:
    raise KeyError("NPZ missing both 'nu' and 'nu_log'")

N = min(N_DRAWS, mu_monkey.shape[0], beta_human.shape[0], sigma_monkey.shape[0], sigma_human.shape[0], L_Rho.shape[0], nu.shape[0])
draw_idx = rng.choice(mu_monkey.shape[0], size=N, replace=False)

# Display grids
pre_ms_vals = np.array(PRE_MS_LIST, dtype=float)
a_vals_grid = pre_ms_vals / float(T_TOTAL_MS)
m_vals_h = np.array([2, 3, 4], dtype=int)
m_vals_m = np.array([2, 3], dtype=int)
q_vals_h = np.array(Q_HUMAN_LIST, dtype=float)
q_vals_m = np.array([float(Q_MONKEY)], dtype=float)

# Determine actual monkey conditions (prefer explicit CSV)
mon_df = None
default_mon_csv = MONKEY_DATA_CSV or os.path.join(PROJECT_ROOT, "data", "monkey_agg_long_FullRun_filtered.csv" if USE_FILTERED_DATA else "monkey_agg_long_FullRun.csv")
try:
    if os.path.exists(default_mon_csv):
        mon_df = pd.read_csv(default_mon_csv)
except Exception:
    mon_df = None

mon_pre_ms_available = None
mon_m_available = None
if MONKEY_CONDITIONS_CSV and os.path.exists(MONKEY_CONDITIONS_CSV):
    try:
        mc = pd.read_csv(MONKEY_CONDITIONS_CSV)
        if {"precue_ms", "load_m"}.issubset(mc.columns):
            mon_pre_ms_available = sorted(pd.unique(mc["precue_ms"].astype(float)).tolist())
            mon_m_available = sorted(pd.unique(mc["load_m"].astype(int)).tolist())
        else:
            mon_pre_ms_available, mon_m_available = None, None
    except Exception:
        mon_pre_ms_available, mon_m_available = None, None

if (mon_pre_ms_available is None) or (mon_m_available is None):
    if mon_df is not None and {"pre_ms", "m"}.issubset(mon_df.columns):
        mon_pre_ms_available = sorted(pd.unique(mon_df["pre_ms"].astype(float)).tolist())
        mon_m_available = sorted(pd.unique(mon_df["m"].astype(int)).tolist())
        mon_m_available = [m for m in mon_m_available if m in {2, 3}]
    else:
        mon_pre_ms_available = [v for v in pre_ms_vals if v in {300.0, 700.0}]
        mon_m_available = [2, 3]

a_vals_monkey = np.array(mon_pre_ms_available, dtype=float) / float(T_TOTAL_MS)
m_vals_m = np.array(mon_m_available, dtype=int)
pre_col_map = {float(v): i for i, v in enumerate(pre_ms_vals.tolist())}
mon_pre_to_col = {float(v): pre_col_map[float(v)] for v in mon_pre_ms_available if float(v) in pre_col_map}

#%%
# Compute species means across draws for PPC
mon_valid = {ci: np.full((N, m_vals_m.size), np.nan, dtype=float) for ci in range(3)}
mon_invalid = {ci: np.full((N, m_vals_m.size), np.nan, dtype=float) for ci in range(3)}
hum70_valid = {ai: np.empty((N, m_vals_h.size), float) for ai in range(a_vals_grid.size)}
hum70_invalid = {ai: np.empty((N, m_vals_h.size), float) for ai in range(a_vals_grid.size)}
hum92_valid = {ai: np.empty((N, m_vals_h.size), float) for ai in range(a_vals_grid.size)}
hum92_invalid = {ai: np.empty((N, m_vals_h.size), float) for ai in range(a_vals_grid.size)}

print(f"Computing PPC for {N} draws…")
for r, t in enumerate(draw_idx):
    mu_m = mu_monkey[t, :]
    mu_h = mu_monkey[t, :] + beta_human[t, :]
    sig_m = sigma_monkey[t, :]
    sig_h = sigma_human[t, :]
    L = L_Rho[t, :, :]

    # Monkeys (q = Q_MONKEY)
    mon_res = compute_cell_means_for_draw(
        species="monkey",
        mu_draw=mu_m, sigma_draw=sig_m, L_Rho_draw=L, nu_draw=nu[t],
        q_vals=q_vals_m, a_vals=a_vals_monkey, m_vals=m_vals_m,
        n_subj=N_SUBJ_MONKEY, predict=predict, rng=rng,
    )
    for ai, pre_ms_val in enumerate(mon_pre_ms_available):
        col = mon_pre_to_col.get(float(pre_ms_val), None)
        if col is not None and col in mon_valid:
            mon_valid[col][r, :] = mon_res[ai]["valid"]
            mon_invalid[col][r, :] = mon_res[ai]["invalid"]

    # Humans q = 0.70 and q = 0.92
    hum70_res = compute_cell_means_for_draw(
        species="human",
        mu_draw=mu_h, sigma_draw=sig_h, L_Rho_draw=L, nu_draw=nu[t],
        q_vals=np.array([0.70], float), a_vals=a_vals_grid, m_vals=m_vals_h,
        n_subj=N_SUBJ_HUMAN, predict=predict, rng=rng,
    )
    for ai in range(a_vals_grid.size):
        hum70_valid[ai][r, :] = hum70_res[ai]["valid"]
        hum70_invalid[ai][r, :] = hum70_res[ai]["invalid"]

    hum92_res = compute_cell_means_for_draw(
        species="human",
        mu_draw=mu_h, sigma_draw=sig_h, L_Rho_draw=L, nu_draw=nu[t],
        q_vals=np.array([0.92], float), a_vals=a_vals_grid, m_vals=m_vals_h,
        n_subj=N_SUBJ_HUMAN, predict=predict, rng=rng,
    )
    for ai in range(a_vals_grid.size):
        hum92_valid[ai][r, :] = hum92_res[ai]["valid"]
        hum92_invalid[ai][r, :] = hum92_res[ai]["invalid"]

print("PPC computation completed")

#%%
from matplotlib.lines import Line2D

# Common labels/colors
label_map = {"valid": "Valid", "invalid": "Invalid"}
colors = {"valid": "#1f77b4", "invalid": "#d62728"}

# Load dataframes
hum_csv = HUMAN_DATA_CSV or os.path.join(PROJECT_ROOT, "data", "human_agg_long_FullRun_filtered.csv" if USE_FILTERED_DATA else "human_agg_long_FullRun.csv")
mon_csv = MONKEY_DATA_CSV or os.path.join(PROJECT_ROOT, "data", "monkey_agg_long_FullRun_filtered.csv" if USE_FILTERED_DATA else "monkey_agg_long_FullRun.csv")
hum_df = pd.read_csv(hum_csv) if os.path.exists(hum_csv) else None
mon_df = pd.read_csv(mon_csv) if os.path.exists(mon_csv) else None

# Pre-cue columns that actually have monkey data (used to gate model plotting)
mon_cols_with_data: set[float] = set()
if mon_df is not None and "pre_ms" in mon_df.columns:
    try:
        mon_cols_with_data = set(map(float, pd.unique(mon_df["pre_ms"].astype(float))))
    except Exception:
        mon_cols_with_data = set()

# Mapping from pre_ms -> list of available loads m in monkey data
mon_m_by_pre: dict[float, list[int]] = {}
if mon_df is not None and {"pre_ms", "m"}.issubset(mon_df.columns):
    try:
        for pre_v in pd.unique(mon_df["pre_ms"].astype(float)):
            sub_m = sorted(map(int, pd.unique(mon_df[mon_df["pre_ms"].astype(float) == float(pre_v)]["m"].astype(int))))
            # keep only loads present in current simulation m grid
            allowed_global = set(m_vals_m.tolist()) if isinstance(m_vals_m, np.ndarray) else set([2, 3])
            mon_m_by_pre[float(pre_v)] = [m for m in sub_m if m in allowed_global]
    except Exception:
        mon_m_by_pre = {}

# ---------- DATA figure (3x3), PNAS large size ----------
fig_data, axs_data = plt.subplots(3, 3, figsize=(6.5, 5.5), sharex=True, sharey=True)

row_defs = [
    ("Rhesus Macaques – q = .92", mon_df, m_vals_m, None),
    ("Humans – q = .70", hum_df[hum_df["q"].round(2) == 0.70] if hum_df is not None else None, m_vals_h, 0.70),
    ("Humans – q = .92", hum_df[hum_df["q"].round(2) == 0.92] if hum_df is not None else None, m_vals_h, 0.92),
]

for r in range(3):
    row_name, df_row, m_vals_row, qval = row_defs[r]
    for c, pre in enumerate(pre_ms_vals):
        ax = axs_data[r, c]
        # Compute subject-mean accuracy and CI from raw data
        if df_row is None:
            ax.set_xticks(list(m_vals_row.astype(int)))
            ax.set_ylim(0.0, 1.0)
            continue
        sub_raw = df_row[(df_row["pre_ms"] == float(pre)) & df_row["label"].isin(["congruent_true", "congruent_false"])].copy()
        if sub_raw.empty:
            ax.set_xticks(list(m_vals_row.astype(int)))
            ax.set_ylim(0.0, 1.0)
            continue
        stats = _data_mean_ci(sub_raw, ["m", "label"]).sort_values(["m", "label"])  # per m
        for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
            g = stats[stats["label"] == lab]
            ax.errorbar(g["m"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=1.6, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
        ax.set_xticks(list(m_vals_row.astype(int)))
        ax.set_ylim(0.0, 1.0)

# Column titles
col_titles = [f"Cue Time – {int(v)} ms" for v in pre_ms_vals]
for c in range(3):
    axs_data[0, c].set_title(col_titles[c])

# Row labels as y-labels on the left column
axs_data[0, 0].set_ylabel("Rhesus Macaques – q = .92")
axs_data[1, 0].set_ylabel("Humans – q = .70")
axs_data[2, 0].set_ylabel("Humans – q = .92")

# Major axis labels
fig_data.supylabel("Accuracy")
fig_data.supxlabel("Load")

# Single legend (top center)
legend_handles = [
    Line2D([0], [0], marker='o', color=colors['valid'], label=label_map['valid'], linestyle='-', linewidth=1.6),
    Line2D([0], [0], marker='o', color=colors['invalid'], label=label_map['invalid'], linestyle='-', linewidth=1.6),
]
fig_data.legend(handles=legend_handles, loc='center left', ncol=1, frameon=False, bbox_to_anchor=(1.005, 0.5))
# leave a smaller right margin for legend (closer to plots)
fig_data.tight_layout(rect=(0, 0, 1.025, 1.025))
out_grid_data = os.path.join(OUT_DIR, f"{PREFIX}_grid_data.png")
fig_data.savefig(out_grid_data, dpi=300)
print(f"Saved: {out_grid_data}")
plt.show()

# ---------- MODEL figure (3x3), PNAS large size ----------
fig_model, axs_model = plt.subplots(3, 3, figsize=(6.5, 5.5), sharex=True, sharey=True)

def _plot_model_cell(ax, mv_draws: np.ndarray, mi_draws: np.ndarray, m_vals_row: np.ndarray):
    if np.all(np.isnan(mv_draws)) or np.all(np.isnan(mi_draws)):
        ax.set_xticks(list(m_vals_row.astype(int)))
        ax.set_ylim(0.0, 1.0)
        return
    mv_m, mv_lo, mv_hi = bands_over_draws(mv_draws)
    mi_m, mi_lo, mi_hi = bands_over_draws(mi_draws)
    ax.errorbar(list(m_vals_row.astype(int)), mv_m, yerr=np.vstack([mv_m - mv_lo, mv_hi - mv_m]), marker="o", linewidth=1.6, color=colors["valid"], label=label_map["valid"], capsize=3)
    ax.errorbar(list(m_vals_row.astype(int)), mi_m, yerr=np.vstack([mi_m - mi_lo, mi_hi - mi_m]), marker="o", linewidth=1.6, color=colors["invalid"], label=label_map["invalid"], capsize=3)
    ax.set_xticks(list(m_vals_row.astype(int)))
    ax.set_ylim(0.0, 1.0)

for c, pre in enumerate(pre_ms_vals):
    # Row 0: Monkeys — only plot where data exist (prefer gating by actual data columns)
    allow_monkey_plot = (float(pre) in mon_cols_with_data) if mon_cols_with_data else (float(pre) in set(mon_pre_ms_available))
    if allow_monkey_plot:
        # Further restrict to loads actually observed for this pre_ms in monkey data
        allowed_m_list = mon_m_by_pre.get(float(pre), list(m_vals_m.astype(int)))
        if len(allowed_m_list) == 0:
            axs_model[0, c].set_xticks(list(m_vals_m.astype(int)))
            axs_model[0, c].set_ylim(0.0, 1.0)
        else:
            idxs = [i for i, m in enumerate(list(m_vals_m.astype(int))) if m in set(allowed_m_list)]
            _plot_model_cell(
                axs_model[0, c],
                mon_valid[c][:, idxs],
                mon_invalid[c][:, idxs],
                np.array(allowed_m_list, dtype=int),
            )
    else:
        # Leave cell without model prediction (keep axes styling consistent)
        axs_model[0, c].set_xticks(list(m_vals_m.astype(int)))
        axs_model[0, c].set_ylim(0.0, 1.0)
    # Row 1: Humans q=0.70 (ai == c)
    _plot_model_cell(axs_model[1, c], hum70_valid[c], hum70_invalid[c], m_vals_h)
    # Row 2: Humans q=0.92 (ai == c)
    _plot_model_cell(axs_model[2, c], hum92_valid[c], hum92_invalid[c], m_vals_h)

# Column titles
for c in range(3):
    axs_model[0, c].set_title(col_titles[c])

# Row labels
axs_model[0, 0].set_ylabel("Rhesus Macaques – q = .92")
axs_model[1, 0].set_ylabel("Humans – q = .70")
axs_model[2, 0].set_ylabel("Humans – q = .92")

# Major labels
fig_model.supylabel("Accuracy")
fig_model.supxlabel("Load")

fig_model.legend(handles=legend_handles, loc='center left', ncol=1, frameon=False, bbox_to_anchor=(1.005, 0.5))
# leave a smaller right margin for legend (closer to plots)
fig_model.tight_layout(rect=(0, 0, 1.025, 1.025))
out_grid_model = os.path.join(OUT_DIR, f"{PREFIX}_grid_model.png")
fig_model.savefig(out_grid_model, dpi=300)
print(f"Saved: {out_grid_model}")
plt.show()

#%%
# Aggregated humans-only plots (same as CLI script)
def agg_over(vals: Dict[int, np.ndarray], target_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    if target_axis == "m":
        stacks_v = []; stacks_i = []
        for ai in range(a_vals_grid.size):
            stacks_v.append(hum70_valid[ai]); stacks_i.append(hum70_invalid[ai])
            stacks_v.append(hum92_valid[ai]); stacks_i.append(hum92_invalid[ai])
        V = np.mean(np.stack(stacks_v, axis=0), axis=0)
        I = np.mean(np.stack(stacks_i, axis=0), axis=0)
        return V, I
    elif target_axis == "pre":
        V_list = []; I_list = []
        for ai in range(a_vals_grid.size):
            V_ai = 0.5 * (np.mean(hum70_valid[ai], axis=1) + np.mean(hum92_valid[ai], axis=1))
            I_ai = 0.5 * (np.mean(hum70_invalid[ai], axis=1) + np.mean(hum92_invalid[ai], axis=1))
            V_list.append(V_ai); I_list.append(I_ai)
        V = np.stack(V_list, axis=1); I = np.stack(I_list, axis=1)
        return V, I
    elif target_axis == "q":
        V70 = np.mean(np.stack([hum70_valid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
        I70 = np.mean(np.stack([hum70_invalid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
        V92 = np.mean(np.stack([hum92_valid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
        I92 = np.mean(np.stack([hum92_invalid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
        V = np.stack([V70, V92], axis=1)
        I = np.stack([I70, I92], axis=1)
        return V, I
    else:
        raise ValueError(target_axis)

# Accuracy vs load (aggregate over pre_ms and q)
V_m, I_m = agg_over(hum70_valid, "m")
Vm, Vm_lo, Vm_hi = bands_over_draws(V_m)
Im, Im_lo, Im_hi = bands_over_draws(I_m)
fig1, (ax1_l, ax1_r) = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
if hum_df is not None:
    stats = data_human_acc_vs_load(hum_df)
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax1_l.errorbar(g["m"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax1_l.set_title("Data"); ax1_l.set_xlabel("Load (m)"); ax1_l.set_ylabel("Accuracy"); ax1_l.set_xticks(list(m_vals_h.astype(int)))
ax1_r.errorbar(m_vals_h, Vm, yerr=np.vstack([Vm - Vm_lo, Vm_hi - Vm]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax1_r.errorbar(m_vals_h, Im, yerr=np.vstack([Im - Im_lo, Im_hi - Im]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax1_r.set_title("Model"); ax1_r.set_xlabel("Load (m)"); ax1_r.legend(frameon=False)
out1 = os.path.join(OUT_DIR, f"{PREFIX}_human_acc_vs_load.png"); fig1.tight_layout(); fig1.savefig(out1, dpi=300); print(f"Saved: {out1}")

# Accuracy vs pre_ms (aggregate over load and q)
V_p, I_p = agg_over(hum70_valid, "pre")
Vp, Vp_lo, Vp_hi = bands_over_draws(V_p)
Ip, Ip_lo, Ip_hi = bands_over_draws(I_p)
fig2, (ax2_l, ax2_r) = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
if hum_df is not None:
    stats = data_human_acc_vs_prems(hum_df)
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax2_l.errorbar(g["pre_ms"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax2_l.set_title("Data"); ax2_l.set_xlabel("Pre-cue timing (ms)"); ax2_l.set_ylabel("Accuracy"); ax2_l.set_xticks(list(pre_ms_vals.astype(int)))
ax2_r.errorbar(pre_ms_vals, Vp, yerr=np.vstack([Vp - Vp_lo, Vp_hi - Vp]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax2_r.errorbar(pre_ms_vals, Ip, yerr=np.vstack([Ip - Ip_lo, Ip_hi - Ip]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax2_r.set_title("Model"); ax2_r.set_xlabel("Pre-cue timing (ms)"); ax2_r.legend(frameon=False)
out2 = os.path.join(OUT_DIR, f"{PREFIX}_human_acc_vs_prems.png"); fig2.tight_layout(); fig2.savefig(out2, dpi=300); print(f"Saved: {out2}")

# Accuracy vs cue reliability q (aggregate over pre_ms and load)
V_q70 = np.mean(np.stack([hum70_valid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
I_q70 = np.mean(np.stack([hum70_invalid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
V_q92 = np.mean(np.stack([hum92_valid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
I_q92 = np.mean(np.stack([hum92_invalid[ai] for ai in range(a_vals_grid.size)], axis=0), axis=(0, 2))
V_q = np.stack([V_q70, V_q92], axis=1)
I_q = np.stack([I_q70, I_q92], axis=1)
Vq, Vq_lo, Vq_hi = bands_over_draws(V_q)
Iq, Iq_lo, Iq_hi = bands_over_draws(I_q)
fig3, (ax3_l, ax3_r) = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
if hum_df is not None:
    stats = data_human_acc_vs_q(hum_df)
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax3_l.errorbar(g["q"], g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax3_l.set_title("Data"); ax3_l.set_xlabel("Cue reliability (q)"); ax3_l.set_ylabel("Accuracy"); ax3_l.set_xticks(Q_HUMAN_LIST)
ax3_r.errorbar(Q_HUMAN_LIST, Vq, yerr=np.vstack([Vq - Vq_lo, Vq_hi - Vq]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax3_r.errorbar(Q_HUMAN_LIST, Iq, yerr=np.vstack([Iq - Iq_lo, Iq_hi - Iq]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax3_r.set_title("Model"); ax3_r.set_xlabel("Cue reliability (q)"); ax3_r.legend(frameon=False)
out3 = os.path.join(OUT_DIR, f"{PREFIX}_human_acc_vs_q.png"); fig3.tight_layout(); fig3.savefig(out3, dpi=300); print(f"Saved: {out3}")




# %%
