#!/usr/bin/env python3
"""
Combined plotting entrypoint that generates all manuscript figures.

Principles:
- Compute shared quantities once (posterior draws, emulator predictions) and reuse.
- One figure per magic cell (#%%) for clean interactive execution.
- Add concise comments/docstrings in non-obvious parts for readability.
"""
#%%
# Combined plotting: posterior summaries/derived (IPC, proportion control)
# and PPC/data comparison figures

import os
import glob
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from joblib import load
from scipy.stats import gaussian_kde

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
# Configuration
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

NPZ_FILE = os.path.join(PROJECT_ROOT, "stan_fits_cmdstan_ma_robust_excl_FullRun", "draws_basic_ma.npz")
EMULATOR_PIPELINE = os.path.join(PROJECT_ROOT, "emulator_artifacts", "train_emu_FullRun", "mlp_emulator.joblib")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_plots")
PREFIX = "FullRun_filtered"

# Posterior-derived config
Q_STAR = 0.92 # reference for IPC computation
PRE_MS = 700.0 # reference for IPC computation
M = 3 # reference for IPC computation
T_TOTAL_MS = 1350.0
N_SUBJECT_SIM = 200 # number of subjects to sample for PPC
SEED = 123

# PPC config
USE_FILTERED_DATA = True
HUMAN_DATA_CSV = None
MONKEY_DATA_CSV = None
MONKEY_CONDITIONS_CSV = os.path.join(PROJECT_ROOT, "data", "monkey_aggregated_2.csv")
PRE_MS_LIST = [300, 700, 1100]
Q_HUMAN_LIST = [0.70, 0.92]
Q_MONKEY = 0.92

os.makedirs(OUT_DIR, exist_ok=True)

print("Config:")
print("  NPZ:", NPZ_FILE)
print("  Emulator:", EMULATOR_PIPELINE)
print("  OUT_DIR:", OUT_DIR)

#%%
# Shared helpers

def flatten_draws(arr: np.ndarray) -> np.ndarray:
    """Flatten (chains, draws, ...) -> (draws, ...), preserving parameter dims.

    If the last two dims form a square (e.g., 2x2 Cholesky), we keep those intact
    and only flatten across the leading chain/draw dimensions.
    """
    if arr.ndim <= 1:
        return arr
    if arr.ndim >= 3 and arr.shape[-1] == arr.shape[-2]:
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return arr.reshape(-1, arr.shape[-1])


def summarize(name: str, x: np.ndarray) -> str:
    """Return a short string with posterior mean and equal-tailed 95% CI."""
    m = float(np.mean(x)); lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{name}: mean={m:.3g}, 95% CI=({lo:.3g}, {hi:.3g})"


def build_emulator_predict_fn(pipeline_path: str):
    """Load scikit-learn pipeline and wrap predict() -> (p_valid, p_invalid).

    The MLP was trained on features:
      [log10_eta, log10_kappa, log10_nu, q, a, m_2, m_3, m_4]
    and outputs logits for (p_valid, p_invalid). We convert logits to
    probabilities via the logistic transform.
    """
    pipe = load(pipeline_path)
    def predict(log10_eta: float, log10_kap: float, log10_nu: float, q: float, a: float, m: int) -> Tuple[float, float]:
        m2 = 1.0 if m == 2 else 0.0
        m3 = 1.0 if m == 3 else 0.0
        m4 = 1.0 if m == 4 else 0.0
        x = np.array([[log10_eta, log10_kap, log10_nu, q, a, m2, m3, m4]], dtype=float)
        z = pipe.predict(x)[0]
        p_valid = float(1.0 / (1.0 + np.exp(-z[0])))
        p_invalid = float(1.0 / (1.0 + np.exp(-z[1])))
        return p_valid, p_invalid
    return predict


def bands_over_draws(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and equal-tailed 95% CI across draws for each element/axis.

    Example: X has shape (n_draws, n_loads) => returns (n_loads,) arrays.
    """
    return np.mean(X, axis=0), np.percentile(X, 2.5, axis=0), np.percentile(X, 97.5, axis=0)


def _data_mean_ci(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    """Subject-mean accuracy with normal-approx 95% CI per group.

    We aggregate per subject within the group (sum y, n), compute subject-level
    accuracy, then compute the mean across subjects. CI is 1.96*SEM.
    """
    work = df[df["label"].isin(["congruent_true", "congruent_false"])].copy()
    agg = work.groupby(["id", *by_cols])[['y', 'n']].sum().reset_index()
    agg['acc'] = agg['y'].astype(float) / agg['n'].clip(lower=1).astype(float)
    stats = agg.groupby(by_cols)['acc'].agg(mean_acc='mean', std_acc='std', N='count').reset_index()
    stats['sem'] = stats['std_acc'] / np.sqrt(np.maximum(stats['N'], 1))
    stats['ci95'] = 1.96 * stats['sem']
    return stats


#%%
# Load emulator pipeline and posterior draws
if not os.path.exists(NPZ_FILE):
    raise FileNotFoundError(f"Could not find draws file: {NPZ_FILE}")
if not os.path.exists(EMULATOR_PIPELINE):
    raise FileNotFoundError(f"Emulator pipeline not found: {EMULATOR_PIPELINE}")

rng = np.random.default_rng(SEED)
d = np.load(NPZ_FILE)
predict = build_emulator_predict_fn(EMULATOR_PIPELINE)

# Extract draws shared by both parts
mu_monkey = flatten_draws(d["mu_monkey"])   # (N,2)
beta_human = flatten_draws(d["beta_human"]) # (N,2)
sigma_monkey = flatten_draws(d["sigma_monkey"])  # (N,2)
sigma_human = flatten_draws(d["sigma_human"])    # (N,2)
if "L_Rho" in d.files:
    L_Rho = flatten_draws(d["L_Rho"])  # (N,2,2)
else:
    # Fallback: assume independence across (log η, log κ) within draw.
    L_Rho = np.tile(np.eye(2, dtype=float), (mu_monkey.shape[0], 1, 1))
if "nu" in d.files:
    nu = flatten_draws(d["nu"]).reshape(-1)
elif "nu_log" in d.files:
    nu = np.exp(flatten_draws(d["nu_log"]).reshape(-1))
else:
    raise KeyError("NPZ missing both 'nu' and 'nu_log'")

# Species means in log space and transformed means
log_eta_monkey = mu_monkey[:, 0]
log_kap_monkey = mu_monkey[:, 1]
log_eta_human = mu_monkey[:, 0] + beta_human[:, 0]
log_kap_human = mu_monkey[:, 1] + beta_human[:, 1]
eta_monkey = np.exp(log_eta_monkey)
kap_monkey = np.exp(log_kap_monkey)
eta_human = np.exp(log_eta_human)
kap_human = np.exp(log_kap_human)

#%%
# Posterior-derived: proportion control and IPC via Monte Carlo
a_cell = float(PRE_MS / T_TOTAL_MS)
q_cell = float(Q_STAR)
m_cell = int(M)

n_draws_post = min(mu_monkey.shape[0], beta_human.shape[0], sigma_monkey.shape[0], sigma_human.shape[0], L_Rho.shape[0], nu.shape[0])
ipc_mon = np.empty(n_draws_post, dtype=float)
ipc_hum = np.empty(n_draws_post, dtype=float)
pctrl_mon_mc = np.empty(n_draws_post, dtype=float)
pctrl_hum_mc = np.empty(n_draws_post, dtype=float)
log10 = np.log10

print(f"Computing IPC and proportion control for {n_draws_post} draws (posterior-derived)...")
for t in range(n_draws_post):
    mu_m = mu_monkey[t, :]
    mu_h = mu_monkey[t, :] + beta_human[t, :]
    L = L_Rho[t, :, :]
    L_eff_m = np.diag(sigma_monkey[t, :]) @ L
    L_eff_h = np.diag(sigma_human[t, :]) @ L

    Zm = rng.standard_normal(size=(N_SUBJECT_SIM, 2))
    Zh = rng.standard_normal(size=(N_SUBJECT_SIM, 2))
    log_ek_m = Zm @ L_eff_m.T + mu_m
    log_ek_h = Zh @ L_eff_h.T + mu_h

    eta_m = np.exp(log_ek_m[:, 0]); kap_m = np.exp(log_ek_m[:, 1])
    eta_h = np.exp(log_ek_h[:, 0]); kap_h = np.exp(log_ek_h[:, 1])

    pctrl_mon_mc[t] = float(np.mean(kap_m / (kap_m + eta_m)))
    pctrl_hum_mc[t] = float(np.mean(kap_h / (kap_h + eta_h)))

    log10_nu = float(log10(max(1e-8, nu[t])))
    acc_m = []
    for jj in range(N_SUBJECT_SIM):
        pv, pi = predict(log10(eta_m[jj]), log10(kap_m[jj]), log10_nu, q_cell, a_cell, m_cell)
        acc_m.append(q_cell * pv + (1 - q_cell) * pi)
    acc_h = []
    for jj in range(N_SUBJECT_SIM):
        pv, pi = predict(log10(eta_h[jj]), log10(kap_h[jj]), log10_nu, q_cell, a_cell, m_cell)
        acc_h.append(q_cell * pv + (1 - q_cell) * pi)
    ipc_mon[t] = float(np.mean(acc_m)) * np.log2(3.0)
    ipc_hum[t] = float(np.mean(acc_h)) * np.log2(3.0)

print("Posterior-derived metrics computed.")

#%%
# Posterior parameter summaries and contrasts (means and 95% CI)
print("\n=== Posterior parameter summaries (means and 95% CI) ===")
print(summarize("eta (monkey)", eta_monkey))
print(summarize("kappa (monkey)", kap_monkey))
print(summarize("eta (human)", eta_human))
print(summarize("kappa (human)", kap_human))

eta_diff = eta_human - eta_monkey
kappa_diff = kap_human - kap_monkey
print(summarize("eta (human - monkey)", eta_diff))
print(summarize("kappa (human - monkey)", kappa_diff))

cap_minus_ctrl_monkey = eta_monkey - kap_monkey
cap_minus_ctrl_human = eta_human - kap_human
print(summarize("capacity minus control (monkey)", cap_minus_ctrl_monkey))
print(summarize("capacity minus control (human)", cap_minus_ctrl_human))

print(summarize("proportion control (monkey)", pctrl_mon_mc))
print(summarize("proportion control (human)", pctrl_hum_mc))
print(summarize("proportion control (human - monkey)", pctrl_hum_mc - pctrl_mon_mc))
print(summarize("IPC (monkey)", ipc_mon))
print(summarize("IPC (human)", ipc_hum))
print(summarize("IPC (human - monkey)", ipc_hum - ipc_mon))

#%%
# PPC precomputation using same draws/predict
pre_ms_vals = np.array(PRE_MS_LIST, dtype=float)
a_vals_grid = pre_ms_vals / float(T_TOTAL_MS)
m_vals_h = np.array([2, 3, 4], dtype=int)
m_vals_m = np.array([2, 3], dtype=int)
q_vals_h = np.array(Q_HUMAN_LIST, dtype=float)
q_vals_m = np.array([float(Q_MONKEY)], dtype=float)

# Determine actual monkey conditions
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

N = min(200, mu_monkey.shape[0], beta_human.shape[0], sigma_monkey.shape[0], sigma_human.shape[0], L_Rho.shape[0], nu.shape[0])
draw_idx = rng.choice(mu_monkey.shape[0], size=N, replace=False)

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

    # Monkeys
    # Sample subjects once per draw to induce realistic spread
    def sample_subjects(mu, sigma, L):
        L_eff = np.diag(sigma) @ L
        z = rng.standard_normal(size=(2, N_SUBJECT_SIM))
        return mu[:, None] + L_eff @ z

    log_ek_m = sample_subjects(mu_m, sig_m, L)
    log_ek_h = sample_subjects(mu_h, sig_h, L)
    log10_nu = float(log10(max(1e-8, nu[t])))

    # Monkeys, per pre_ms and m
    for ai, pre_ms_val in enumerate(mon_pre_ms_available):
        a = float(pre_ms_val / T_TOTAL_MS)
        mv = []
        mi = []
        for j in range(N_SUBJECT_SIM):
            eta = float(np.exp(log_ek_m[0, j])); kap = float(np.exp(log_ek_m[1, j]))
            pv, pi = predict(log10(eta), log10(kap), log10_nu, float(Q_MONKEY), a, int(2))  # m placeholder; averaged below
            mv.append(pv); mi.append(pi)
        # Repeat per-load by requerying with correct m while keeping subject set
        mv = np.array(mv, float); mi = np.array(mi, float)  # for shape consistency
        # Fill with per-load predictions
        for li, m_val in enumerate(list(m_vals_m.astype(int))):
            mv_m, mi_m = [], []
            for j in range(N_SUBJECT_SIM):
                eta = float(np.exp(log_ek_m[0, j])); kap = float(np.exp(log_ek_m[1, j]))
                pv, pi = predict(log10(eta), log10(kap), log10_nu, float(Q_MONKEY), a, int(m_val))
                mv_m.append(pv); mi_m.append(pi)
            col = mon_pre_to_col.get(float(pre_ms_val), None)
            if col is not None and col in mon_valid:
                mon_valid[col][r, li] = float(np.mean(mv_m))
                mon_invalid[col][r, li] = float(np.mean(mi_m))

    # Humans q = 0.70 and q = 0.92
    for qi, qv in enumerate([0.70, 0.92]):
        for ai, a in enumerate(a_vals_grid.tolist()):
            v, i_ = [], []
            for j in range(N_SUBJECT_SIM):
                eta = float(np.exp(log_ek_h[0, j])); kap = float(np.exp(log_ek_h[1, j]))
                pv, pi = predict(log10(eta), log10(kap), log10_nu, float(qv), float(a), int(2))
                v.append(pv); i_.append(pi)
            v = np.array(v, float); i_ = np.array(i_, float)
            # per-load
            for li, m_val in enumerate(list(m_vals_h.astype(int))):
                v_m, i_m = [], []
                for j in range(N_SUBJECT_SIM):
                    eta = float(np.exp(log_ek_h[0, j])); kap = float(np.exp(log_ek_h[1, j]))
                    pv, pi = predict(log10(eta), log10(kap), log10_nu, float(qv), float(a), int(m_val))
                    v_m.append(pv); i_m.append(pi)
                (hum70_valid if qv==0.70 else hum92_valid)[ai][r, li] = float(np.mean(v_m))
                (hum70_invalid if qv==0.70 else hum92_invalid)[ai][r, li] = float(np.mean(i_m))

print("PPC computation completed.")

#%%
#%%
# Figure: Posterior 2D contours (η vs κ) by species
print("Generating posterior-derived plots…")

# 2D HDI contours (η vs κ)
def kde_hdi_levels(x, y, xlim, ylim, gridsize=400, masses=(0.5, 0.8, 0.95), bw_factor=None):
    xi = np.linspace(xlim[0], xlim[1], gridsize)
    yi = np.linspace(ylim[0], ylim[1], gridsize)
    xx, yy = np.meshgrid(xi, yi, indexing="xy")
    kde = gaussian_kde(np.vstack([x, y]))
    if bw_factor is not None:
        kde.set_bandwidth(kde.factor * bw_factor)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    dx = (xi[1] - xi[0]); dy = (yi[1] - yi[0])
    flat = zz.ravel(); order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) * dx * dy
    thresholds = []
    for m in masses:
        idx = int(np.searchsorted(csum, m, side="left"))
        idx = int(np.clip(idx, 0, len(order) - 1))
        thresholds.append(float(flat[order[idx]]))
    return xx, yy, zz, thresholds

xlim_raw = [float(np.min(np.concatenate([eta_monkey, eta_human]))) - 0.1 * float(np.std(np.concatenate([eta_monkey, eta_human]))),
            float(np.max(np.concatenate([eta_monkey, eta_human]))) + 0.1 * float(np.std(np.concatenate([eta_monkey, eta_human])))]
ylim_raw = [float(np.min(np.concatenate([kap_monkey, kap_human]))) - 0.1 * float(np.std(np.concatenate([kap_monkey, kap_human]))),
            float(np.max(np.concatenate([kap_monkey, kap_human]))) + 0.1 * float(np.std(np.concatenate([kap_monkey, kap_human])))]

Xm, Ym, Zm, (thr50_m, thr80_m, thr95_m) = kde_hdi_levels(eta_monkey, kap_monkey, xlim_raw, ylim_raw)
Xh, Yh, Zh, (thr50_h, thr80_h, thr95_h) = kde_hdi_levels(eta_human,  kap_human,  xlim_raw, ylim_raw)

fig4, ax4 = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi=400)
ax4.contourf(Xm, Ym, Zm, levels=[thr95_m, float(Zm.max())], colors=["#6baed6"], alpha=0.28)
ax4.contour (Xm, Ym, Zm, levels=[thr95_m], colors=["#2c7fb8"], linewidths=[2.0])
ax4.contour (Xm, Ym, Zm, levels=[thr80_m], colors=["#2c7fb8"], linewidths=[1.4], linestyles="--")
ax4.contour (Xm, Ym, Zm, levels=[thr50_m], colors=["#2c7fb8"], linewidths=[1.2], linestyles="--")
ax4.contourf(Xh, Yh, Zh, levels=[thr95_h, float(Zh.max())], colors=["#fdae6b"], alpha=0.28)
ax4.contour (Xh, Yh, Zh, levels=[thr95_h], colors=["#e6550d"], linewidths=[2.0])
ax4.contour (Xh, Yh, Zh, levels=[thr80_h], colors=["#e6550d"], linewidths=[1.4], linestyles="--")
ax4.contour (Xh, Yh, Zh, levels=[thr50_h], colors=["#e6550d"], linewidths=[1.2], linestyles="--")
ax4.set_xlabel("Capacity η"); ax4.set_ylabel("Control κ")
ax4.grid(False)

all_eta = np.concatenate([eta_monkey, eta_human]); all_kap = np.concatenate([kap_monkey, kap_human])
x_lo, x_hi = np.percentile(all_eta, [0.5, 99.5]); y_lo, y_hi = np.percentile(all_kap, [0.5, 99.5])
xr = float(x_hi - x_lo); yr = float(y_hi - y_lo)
ax4.set_xlim(float(y_lo - 0.10 * yr), float(x_hi + 0.03 * xr))
ax4.set_ylim(float(y_lo - 0.10 * yr), float(y_hi + 0.10 * yr))

legend_elems = [
    mpatches.Patch(facecolor=mcolors.to_rgba("#fdae6b", 0.35), edgecolor=mcolors.to_rgba("#e6550d", 0.5), linewidth=1.0, label="Humans"),
    mpatches.Patch(facecolor=mcolors.to_rgba("#6baed6", 0.35), edgecolor=mcolors.to_rgba("#2c7fb8", 0.5), linewidth=1.0, label="Rhesus Macaques"),
]
ax4.legend(handles=legend_elems, loc="upper left", frameon=False, handletextpad=0.12, labelspacing=0.15, handlelength=0.8, handleheight=0.45)
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, f"{PREFIX}_2d_overlay_HDI.pdf"), dpi=400, bbox_inches="tight")
print(os.path.join(OUT_DIR, f"{PREFIX}_2d_overlay_HDI.pdf"))

#%%
# Figure: Proportion control histograms by species
fig_pc, ax = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi=400)
ax.hist(pctrl_mon_mc, bins=60, range=(0, 1), density=True, alpha=0.6, color="#6baed6", label="Rhesus Macaques")
ax.hist(pctrl_hum_mc, bins=60, range=(0, 1), density=True, alpha=0.6, color="#fdae6b", label="Humans")
ax.set_ylabel("Density"); ax.set_xlabel("Proportion Control (" + r"$\frac{\kappa}{\kappa + \eta}$" + ")"); ax.set_xlim(0, 1); ax.grid(False)
ax.legend(handles=[mpatches.Patch(facecolor=mcolors.to_rgba("#fdae6b", 0.35), edgecolor=mcolors.to_rgba("#e6550d", 0.5), linewidth=1.0, label="Humans"),
                  mpatches.Patch(facecolor=mcolors.to_rgba("#6baed6", 0.35), edgecolor=mcolors.to_rgba("#2c7fb8", 0.5), linewidth=1.0, label="Rhesus Macaques")],
          loc="upper right", frameon=False, handletextpad=0.4, labelspacing=0.3, handlelength=0.8, handleheight=0.6, fontsize=10, ncol=1)
fig_pc.tight_layout();
path_pctrl = os.path.join(OUT_DIR, f"{PREFIX}_pctrl.pdf")
fig_pc.savefig(path_pctrl, dpi=400, bbox_inches="tight")
print(path_pctrl)

#%%
# Figure: Information-Processing Capability (IPC) histograms by species
fig_ipc, ax = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi=400)
ax.hist(ipc_mon, bins=60, density=True, alpha=0.6, color="#6baed6", label="Rhesus Macaques")
ax.hist(ipc_hum, bins=60, density=True, alpha=0.6, color="#fdae6b", label="Humans")
ax.set_ylabel("Density"); ax.set_xlabel("Information-Processing Capability (Bits)"); ax.grid(False); ax.legend(frameon=False, loc="upper left")
ax.set_xlim(0, 1.58)
fig_ipc.tight_layout();
path_ipc = os.path.join(OUT_DIR, f"{PREFIX}_ipc.pdf")
fig_ipc.savefig(path_ipc, dpi=400, bbox_inches="tight")
print(path_ipc)

#%%
#%%
# Figure: DATA grid (3x3) – Accuracy by Load across pre-cue timings and q
print("Generating PPC and data comparison plots…")

# Load dataframes
hum_csv = HUMAN_DATA_CSV or os.path.join(PROJECT_ROOT, "data", "human_agg_long_FullRun_filtered.csv" if USE_FILTERED_DATA else "human_agg_long_FullRun.csv")
mon_csv = MONKEY_DATA_CSV or os.path.join(PROJECT_ROOT, "data", "monkey_agg_long_FullRun_filtered.csv" if USE_FILTERED_DATA else "monkey_agg_long_FullRun.csv")
hum_df = pd.read_csv(hum_csv) if os.path.exists(hum_csv) else None
mon_df = pd.read_csv(mon_csv) if os.path.exists(mon_csv) else None

label_map = {"valid": "Valid", "invalid": "Invalid"}
colors = {"valid": "#1f77b4", "invalid": "#d62728"}

fig_data, axs_data = plt.subplots(3, 3, figsize=(6.5, 5.5), sharex=True, sharey=True)
row_defs = [
    ("Rhesus Macaques – q = .92", mon_df, m_vals_m, None),
    ("Humans – q = .70", hum_df[hum_df["q"].round(2) == 0.70] if hum_df is not None else None, m_vals_h, 0.70),
    ("Humans – q = .92", hum_df[hum_df["q"].round(2) == 0.92] if hum_df is not None else None, m_vals_h, 0.92),
]

for r in range(3):
    _, df_row, m_vals_row, _ = row_defs[r]
    for c, pre in enumerate(pre_ms_vals):
        ax = axs_data[r, c]
        if df_row is None:
            ax.set_xticks(list(m_vals_row.astype(int))); ax.set_ylim(0.0, 1.0); continue
        sub_raw = df_row[(df_row["pre_ms"] == float(pre)) & df_row["label"].isin(["congruent_true", "congruent_false"])].copy()
        if sub_raw.empty:
            ax.set_xticks(list(m_vals_row.astype(int))); ax.set_ylim(0.0, 1.0); continue
        stats = _data_mean_ci(sub_raw, ["m", "label"]).sort_values(["m", "label"])  # per m
        for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
            g = stats[stats["label"] == lab]
            ax.errorbar(g["m"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=1.6, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
        ax.set_xticks(list(m_vals_row.astype(int))); ax.set_ylim(0.0, 1.0)

col_titles = [f"Cue Time – {int(v)} ms" for v in pre_ms_vals]
for c in range(3):
    axs_data[0, c].set_title(col_titles[c])
axs_data[0, 0].set_ylabel("Rhesus Macaques – q = .92")
axs_data[1, 0].set_ylabel("Humans – q = .70")
axs_data[2, 0].set_ylabel("Humans – q = .92")
fig_data.supylabel("Accuracy"); fig_data.supxlabel("Load")

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color=colors['valid'], label=label_map['valid'], linestyle='-', linewidth=1.6),
    Line2D([0], [0], marker='o', color=colors['invalid'], label=label_map['invalid'], linestyle='-', linewidth=1.6),
]
fig_data.legend(handles=legend_handles, loc='center left', ncol=1, frameon=False, bbox_to_anchor=(1.005, 0.5))
fig_data.tight_layout(rect=(0, 0, 1.025, 1.025))
path_grid_data = os.path.join(OUT_DIR, f"ppc_{PREFIX}_grid_data.png")
fig_data.savefig(path_grid_data, dpi=300)
print(path_grid_data)

#%%
# Figure: MODEL grid (3x3) – Posterior predictive means by Load across pre-cue timings and q
fig_model, axs_model = plt.subplots(3, 3, figsize=(6.5, 5.5), sharex=True, sharey=True)

def _plot_model_cell(ax, mv_draws: np.ndarray, mi_draws: np.ndarray, m_vals_row: np.ndarray):
    if np.all(np.isnan(mv_draws)) or np.all(np.isnan(mi_draws)):
        ax.set_xticks(list(m_vals_row.astype(int))); ax.set_ylim(0.0, 1.0); return
    mv_m, mv_lo, mv_hi = bands_over_draws(mv_draws)
    mi_m, mi_lo, mi_hi = bands_over_draws(mi_draws)
    ax.errorbar(list(m_vals_row.astype(int)), mv_m, yerr=np.vstack([mv_m - mv_lo, mv_hi - mv_m]), marker="o", linewidth=1.6, color=colors["valid"], label=label_map["valid"], capsize=3)
    ax.errorbar(list(m_vals_row.astype(int)), mi_m, yerr=np.vstack([mi_m - mi_lo, mi_hi - mi_m]), marker="o", linewidth=1.6, color=colors["invalid"], label=label_map["invalid"], capsize=3)
    ax.set_xticks(list(m_vals_row.astype(int))); ax.set_ylim(0.0, 1.0)

for c, pre in enumerate(pre_ms_vals):
    allow_monkey_plot = (float(pre) in set(mon_pre_ms_available))
    if allow_monkey_plot:
        allowed_m_list = mon_m_available
        if len(allowed_m_list) == 0:
            axs_model[0, c].set_xticks(list(m_vals_m.astype(int))); axs_model[0, c].set_ylim(0.0, 1.0)
        else:
            idxs = [i for i, m in enumerate(list(m_vals_m.astype(int))) if m in set(allowed_m_list)]
            _plot_model_cell(axs_model[0, c], mon_valid[c][:, idxs], mon_invalid[c][:, idxs], np.array(allowed_m_list, dtype=int))
    else:
        axs_model[0, c].set_xticks(list(m_vals_m.astype(int))); axs_model[0, c].set_ylim(0.0, 1.0)
    _plot_model_cell(axs_model[1, c], hum70_valid[c], hum70_invalid[c], m_vals_h)
    _plot_model_cell(axs_model[2, c], hum92_valid[c], hum92_invalid[c], m_vals_h)

for c in range(3):
    axs_model[0, c].set_title(col_titles[c])
axs_model[0, 0].set_ylabel("Rhesus Macaques – q = .92")
axs_model[1, 0].set_ylabel("Humans – q = .70")
axs_model[2, 0].set_ylabel("Humans – q = .92")
fig_model.supylabel("Accuracy"); fig_model.supxlabel("Load")
fig_model.legend(handles=legend_handles, loc='center left', ncol=1, frameon=False, bbox_to_anchor=(1.005, 0.5))
fig_model.tight_layout(rect=(0, 0, 1.025, 1.025))
path_grid_model = os.path.join(OUT_DIR, f"ppc_{PREFIX}_grid_model.png")
fig_model.savefig(path_grid_model, dpi=300)
print(path_grid_model)

# Aggregated humans-only plots
def agg_over(target_axis: str) -> Tuple[np.ndarray, np.ndarray]:
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
        V = np.stack([V70, V92], axis=1); I = np.stack([I70, I92], axis=1)
        return V, I
    else:
        raise ValueError(target_axis)

#%%
# Figure: Humans – Accuracy vs Load (Data vs Model)
V_m, I_m = agg_over("m")
Vm, Vm_lo, Vm_hi = bands_over_draws(V_m)
Im, Im_lo, Im_hi = bands_over_draws(I_m)
fig1, (ax1_l, ax1_r) = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
if hum_df is not None:
    stats = _data_mean_ci(hum_df, ["m", "label"]).sort_values(["m", "label"])  # per m
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax1_l.errorbar(g["m"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax1_l.set_title("Data"); ax1_l.set_xlabel("Load (m)"); ax1_l.set_ylabel("Accuracy"); ax1_l.set_xticks(list(m_vals_h.astype(int)))
ax1_r.errorbar(m_vals_h, Vm, yerr=np.vstack([Vm - Vm_lo, Vm_hi - Vm]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax1_r.errorbar(m_vals_h, Im, yerr=np.vstack([Im - Im_lo, Im_hi - Im]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax1_r.set_title("Model"); ax1_r.set_xlabel("Load (m)"); ax1_r.legend(frameon=False)
fig1.tight_layout();
path_vs_load = os.path.join(OUT_DIR, f"ppc_{PREFIX}_human_acc_vs_load.png")
fig1.savefig(path_vs_load, dpi=300)
print(path_vs_load)

#%%
# Figure: Humans – Accuracy vs Pre-cue timing (Data vs Model)
V_p, I_p = agg_over("pre")
Vp, Vp_lo, Vp_hi = bands_over_draws(V_p)
Ip, Ip_lo, Ip_hi = bands_over_draws(I_p)
fig2, (ax2_l, ax2_r) = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
if hum_df is not None:
    stats = _data_mean_ci(hum_df, ["pre_ms", "label"]).sort_values(["pre_ms", "label"])  # per pre_ms
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax2_l.errorbar(g["pre_ms"].astype(int), g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax2_l.set_title("Data"); ax2_l.set_xlabel("Pre-cue timing (ms)"); ax2_l.set_ylabel("Accuracy"); ax2_l.set_xticks(list(pre_ms_vals.astype(int)))
ax2_r.errorbar(pre_ms_vals, Vp, yerr=np.vstack([Vp - Vp_lo, Vp_hi - Vp]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax2_r.errorbar(pre_ms_vals, Ip, yerr=np.vstack([Ip - Ip_lo, Ip_hi - Ip]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax2_r.set_title("Model"); ax2_r.set_xlabel("Pre-cue timing (ms)"); ax2_r.legend(frameon=False)
fig2.tight_layout();
path_vs_pre = os.path.join(OUT_DIR, f"ppc_{PREFIX}_human_acc_vs_prems.png")
fig2.savefig(path_vs_pre, dpi=300)
print(path_vs_pre)

#%%
# Figure: Humans – Accuracy vs Cue reliability q (Data vs Model)
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
    stats = _data_mean_ci(hum_df, ["q", "label"]).sort_values(["q", "label"])  # per q
    for lab, color in [("congruent_true", colors["valid"]), ("congruent_false", colors["invalid"])]:
        g = stats[stats["label"] == lab]
        ax3_l.errorbar(g["q"], g["mean_acc"], yerr=g["ci95"], marker="o", linewidth=2, color=color, label=label_map["valid"] if lab=="congruent_true" else label_map["invalid"], capsize=3)
ax3_l.set_title("Data"); ax3_l.set_xlabel("Cue reliability (q)"); ax3_l.set_ylabel("Accuracy"); ax3_l.set_xticks(Q_HUMAN_LIST)
ax3_r.errorbar(Q_HUMAN_LIST, Vq, yerr=np.vstack([Vq - Vq_lo, Vq_hi - Vq]), marker="o", linewidth=2, color=colors["valid"], label=label_map["valid"], capsize=3)
ax3_r.errorbar(Q_HUMAN_LIST, Iq, yerr=np.vstack([Iq - Iq_lo, Iq_hi - Iq]), marker="o", linewidth=2, color=colors["invalid"], label=label_map["invalid"], capsize=3)
ax3_r.set_title("Model"); ax3_r.set_xlabel("Cue reliability (q)"); ax3_r.legend(frameon=False)
fig3.tight_layout();
path_vs_q = os.path.join(OUT_DIR, f"ppc_{PREFIX}_human_acc_vs_q.png")
fig3.savefig(path_vs_q, dpi=300)
print(path_vs_q)


