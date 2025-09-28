
#%%
# Setup and imports
import argparse
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
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
# Configuration - modify these as needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ_FILE = os.path.join(PROJECT_ROOT, "stan_fits_cmdstan_ma_robust_excl_FullRun", "draws_basic_ma.npz")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_plots")
PREFIX = "posterior_ma_FullRun_filtered_means"
EMULATOR_JSON = os.path.join(PROJECT_ROOT, "emulator_artifacts", "train_emu_FullRun", "emulator_for_stan.json")
Q_STAR = 0.92
PRE_MS = 700.0
M = 3
T_TOTAL_MS = 1350.0
N_SUBJECT_SIM = 200
SEED = 123
# Plot configuration options

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")

print(f"Configuration:")
print(f"  NPZ file: {NPZ_FILE}")
print(f"  Output dir: {OUT_DIR}")
print(f"  Prefix: {PREFIX}")
print(f"  Emulator: {EMULATOR_JSON}")

#%%
# Helper functions
def flatten_draws(arr: np.ndarray) -> np.ndarray:
    """Flatten chains/draws into one axis while preserving parameter dims."""
    if arr.ndim <= 1:
        return arr
    # If last two dims look like a square matrix (e.g., 2x2), keep them
    if arr.ndim >= 3 and arr.shape[-1] == arr.shape[-2]:
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    # Otherwise keep only the last param dim
    return arr.reshape(-1, arr.shape[-1])

def summarize(name: str, x: np.ndarray) -> str:
    m = float(np.mean(x))
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{name}: mean={m:.3g}, 95% CI=({lo:.3g}, {hi:.3g})"

#%%
# Load data and emulator
if not os.path.exists(NPZ_FILE):
    raise FileNotFoundError(f"Could not find draws file: {NPZ_FILE}")

data = np.load(NPZ_FILE)
print(f"Loaded NPZ file with keys: {list(data.keys())}")

# Load emulator
if not os.path.exists(EMULATOR_JSON):
    raise FileNotFoundError(f"Emulator JSON not found at {EMULATOR_JSON}")
with open(EMULATOR_JSON, "r") as f:
    emu = json.load(f)

D = int(emu["D"]); H1 = int(emu["H1"]); H2 = int(emu["H2"]); O = int(emu["O"])
x_mean = np.array(emu["x_mean"], dtype=float)
x_scale = np.array(emu["x_scale"], dtype=float)
W1 = np.array(emu["W1"], dtype=float)
b1 = np.array(emu["b1"], dtype=float)
W2 = np.array(emu["W2"], dtype=float)
b2 = np.array(emu["b2"], dtype=float)
W3 = np.array(emu["W3"], dtype=float)
b3 = np.array(emu["b3"], dtype=float)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def mlp_predict_prob(log10_eta: float, log10_kap: float, log10_nu: float, q: float, a: float, m: int) -> tuple[float, float]:
    m2 = 1.0 if m == 2 else 0.0
    m3 = 1.0 if m == 3 else 0.0
    m4 = 1.0 if m == 4 else 0.0
    x = np.array([log10_eta, log10_kap, log10_nu, q, a, m2, m3, m4], dtype=float)
    z = (x - x_mean) / x_scale
    h1 = relu(z @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    out = h2 @ W3 + b3
    p_valid = float(sigmoid(out[0]))
    p_invalid = float(sigmoid(out[1]))
    return p_valid, p_invalid

print("Emulator loaded successfully")

#%%
# Extract and flatten parameters
# Required group/location parameters
mu_monkey = data["mu_monkey"]  # (draws, 2) or (chains, draws, 2)
beta_human = data["beta_human"]
# Species-specific subject SDs
sigma_monkey = data["sigma_monkey"]
sigma_human = data["sigma_human"]
# Correlation (Cholesky) across log eta/kappa
if "L_Rho" in data.files:
    L_Rho = data["L_Rho"]  # (draws, 2, 2) or (chains, draws, 2, 2)
else:
    # Fallback: identity per draw
    draws_guess = mu_monkey.shape[0]
    L_Rho = np.tile(np.eye(2, dtype=float), (draws_guess, 1, 1))

# nu may be stored either as exp(nu_log) or as nu_log
if "nu" in data.files:
    nu = data["nu"]
elif "nu_log" in data.files:
    nu = np.exp(data["nu_log"])  # convert to raw scale
else:
    raise KeyError("NPZ missing both 'nu' and 'nu_log'")

# Flatten draws across chains if present
mu_monkey = flatten_draws(mu_monkey)
beta_human = flatten_draws(beta_human)
sigma_monkey = flatten_draws(sigma_monkey)
sigma_human = flatten_draws(sigma_human)
nu = flatten_draws(nu).reshape(-1)
L_Rho = flatten_draws(L_Rho)

# Group means in log space
log_eta_monkey = mu_monkey[:, 0]
log_kap_monkey = mu_monkey[:, 1]
log_eta_human = mu_monkey[:, 0] + beta_human[:, 0]
log_kap_human = mu_monkey[:, 1] + beta_human[:, 1]

# *** KEY CHANGE: Transform to MEANS on original scale using log-normal mean correction ***
# E[exp(Z)] = exp(μ + 0.5*σ²) where Z ~ N(μ, σ²)


eta_monkey = np.exp(log_eta_monkey)
kap_monkey = np.exp(log_kap_monkey)
eta_human = np.exp(log_eta_human)
kap_human = np.exp(log_kap_human)

# Optional robust-model parameters
tau_subj = data["tau_subj"] if "tau_subj" in data.files else None
epsilon = data["epsilon"] if "epsilon" in data.files else None

print("Parameters extracted and flattened")

#%%
# Monte Carlo computation for proportion control and IPC
rng = np.random.default_rng(SEED)
a_cell = float(PRE_MS / T_TOTAL_MS)
q_cell = float(Q_STAR)
m_cell = int(M)

# We approximate species mean accuracy and proportion control by Monte Carlo over subject effects per posterior draw
n_draws = min(
    mu_monkey.shape[0], beta_human.shape[0], sigma_monkey.shape[0], sigma_human.shape[0], L_Rho.shape[0], nu.shape[0]
)

# Arrays to store Monte Carlo results
ipc_mon = np.empty(n_draws, dtype=float)
ipc_hum = np.empty(n_draws, dtype=float)
pctrl_mon_mc = np.empty(n_draws, dtype=float)  # Monte Carlo proportion control for monkeys
pctrl_hum_mc = np.empty(n_draws, dtype=float)  # Monte Carlo proportion control for humans
log10 = np.log10

print(f"Computing IPC and proportion control for {n_draws} draws...")
for t in range(n_draws):
    # Species means on log scale
    mu_m = mu_monkey[t, :]
    mu_h = mu_monkey[t, :] + beta_human[t, :]
    
    # Subject scatter via cholesky
    L = L_Rho[t, :, :]
    L_eff_m = np.diag(sigma_monkey[t, :]) @ L
    L_eff_h = np.diag(sigma_human[t, :]) @ L
    
    # Sample subjects
    Zm = rng.standard_normal(size=(N_SUBJECT_SIM, 2))
    Zh = rng.standard_normal(size=(N_SUBJECT_SIM, 2))
    log_ek_m = Zm @ L_eff_m.T + mu_m  # (N,2)
    log_ek_h = Zh @ L_eff_h.T + mu_h
    
    # Transform to original scale for this draw
    eta_m = np.exp(log_ek_m[:, 0])
    kap_m = np.exp(log_ek_m[:, 1])
    eta_h = np.exp(log_ek_h[:, 0])
    kap_h = np.exp(log_ek_h[:, 1])
    
    # Compute proportion control for this draw (Monte Carlo mean)
    pctrl_mon_mc[t] = float(np.mean(kap_m / (kap_m + eta_m)))
    pctrl_hum_mc[t] = float(np.mean(kap_h / (kap_h + eta_h)))
    
    # Emulator predictions and mixture by q for IPC
    log10_nu = float(log10(max(1e-8, nu[t])))
    
    # Monkeys
    acc_m = []
    for jj in range(N_SUBJECT_SIM):
        eta = float(eta_m[jj]); kap = float(kap_m[jj])
        p_val, p_inv = mlp_predict_prob(log10(eta), log10(kap), log10_nu, q_cell, a_cell, m_cell)
        acc_m.append(q_cell * p_val + (1 - q_cell) * p_inv)
    
    # Humans
    acc_h = []
    for jj in range(N_SUBJECT_SIM):
        eta = float(eta_h[jj]); kap = float(kap_h[jj])
        p_val, p_inv = mlp_predict_prob(log10(eta), log10(kap), log10_nu, q_cell, a_cell, m_cell)
        acc_h.append(q_cell * p_val + (1 - q_cell) * p_inv)
    
    Rbar_m = float(np.mean(acc_m))
    Rbar_h = float(np.mean(acc_h))
    ipc_mon[t] = Rbar_m * np.log2(3.0)
    ipc_hum[t] = Rbar_h * np.log2(3.0)

print("IPC and proportion control computation completed")

#%%
# Print summary statistics
print("\n=== Parameter Summaries (MEAN-CORRECTED) ===")
print(summarize("eta (monkey) - MEAN", eta_monkey))
print(summarize("kappa (monkey) - MEAN", kap_monkey))
print(summarize("eta (human) - MEAN", eta_human))
print(summarize("kappa (human) - MEAN", kap_human))

# Compute and print summary for eta_human - eta_monkey
eta_diff = eta_human - eta_monkey
print(summarize("eta (human - monkey) - MEAN DIFF", eta_diff))
kappa_diff = kap_human - kap_monkey
print(summarize("kappa (human - monkey) - MEAN DIFF", kappa_diff))

# Compute "capacity minus control" for both species
cap_minus_ctrl_monkey = eta_monkey - kap_monkey
cap_minus_ctrl_human = eta_human - kap_human
print(summarize("capacity minus control (monkey) - MEAN", cap_minus_ctrl_monkey))
print(summarize("capacity minus control (human) - MEAN", cap_minus_ctrl_human))

# *** NEW: Proportion control computed via Monte Carlo ***
print(summarize("proportion control (monkey) - MC MEAN", pctrl_mon_mc))
print(summarize("proportion control (human) - MC MEAN", pctrl_hum_mc))
diff_pctrl_mc = pctrl_hum_mc - pctrl_mon_mc
print(summarize("proportion control (human - monkey) - MC DIFF", diff_pctrl_mc))

print(summarize("IPC (monkey)", ipc_mon))
print(summarize("IPC (human)", ipc_hum))
ipc_diff = ipc_hum - ipc_mon
print(summarize("IPC (human - monkey)", ipc_diff))

#%%
# Multi-level (50/80/95%) HDIs via KDE (filled 95%) in raw scale - FIRST PLOT

# Toggle: overlay raw posterior samples on the contour plot
OVERLAY_SAMPLES_ON_OVERLAY = False
SAMPLES_POINT_SIZE = 6
SAMPLES_ALPHA = 0.25

from scipy.stats import gaussian_kde

def kde_hdi_levels(x, y, xlim, ylim, gridsize=400, masses=(0.5, 0.8, 0.95), bw_factor=None):
    """Calculate HDI levels by finding smallest regions containing given probability mass"""
    xi = np.linspace(xlim[0], xlim[1], gridsize)
    yi = np.linspace(ylim[0], ylim[1], gridsize)
    xx, yy = np.meshgrid(xi, yi, indexing="xy")
    kde = gaussian_kde(np.vstack([x, y]))
    if bw_factor is not None:
        kde.set_bandwidth(kde.factor * bw_factor)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # density thresholds for each mass level
    dx = (xi[1] - xi[0]); dy = (yi[1] - yi[0])
    flat = zz.ravel(); order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) * dx * dy
    thresholds = []
    for m in masses:
        idx = int(np.searchsorted(csum, m, side="left"))
        # Clip index to valid bounds
        idx = int(np.clip(idx, 0, len(order) - 1))
        thresholds.append(float(flat[order[idx]]))
    return xx, yy, zz, thresholds

# Extents
xlim_raw = [
    float(np.min(np.concatenate([eta_monkey, eta_human]))) - 0.1 * float(np.std(np.concatenate([eta_monkey, eta_human]))),
    float(np.max(np.concatenate([eta_monkey, eta_human]))) + 0.1 * float(np.std(np.concatenate([eta_monkey, eta_human]))),
]
ylim_raw = [
    float(np.min(np.concatenate([kap_monkey, kap_human]))) - 0.1 * float(np.std(np.concatenate([kap_monkey, kap_human]))),
    float(np.max(np.concatenate([kap_monkey, kap_human]))) + 0.1 * float(np.std(np.concatenate([kap_monkey, kap_human]))),
]

masses = (0.5, 0.8, 0.95)
print("Generating HDI plot...")
Xm, Ym, Zm, (thr50_m, thr80_m, thr95_m) = kde_hdi_levels(eta_monkey, kap_monkey, xlim_raw, ylim_raw, gridsize=400, masses=masses)
Xh, Yh, Zh, (thr50_h, thr80_h, thr95_h) = kde_hdi_levels(eta_human,  kap_human,  xlim_raw, ylim_raw, gridsize=400, masses=masses)
plot_type = "HDI"

fig4, ax4 = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi = 400)

# Monkey: 95% region fill (skip when overlaying samples) + 95/80/50 contours
if not OVERLAY_SAMPLES_ON_OVERLAY:
    ax4.contourf(Xm, Ym, Zm, levels=[thr95_m, float(Zm.max())], colors=["#6baed6"], alpha=0.28)
ax4.contour (Xm, Ym, Zm, levels=[thr95_m], colors=["#2c7fb8"], linewidths=[2.0], zorder=3)
ax4.contour (Xm, Ym, Zm, levels=[thr80_m], colors=["#2c7fb8"], linewidths=[1.4], linestyles="--", zorder=3)
ax4.contour (Xm, Ym, Zm, levels=[thr50_m], colors=["#2c7fb8"], linewidths=[1.2], linestyles="--", zorder=3)

# Human: 95% region fill (skip when overlaying samples) + 95/80/50 contours
if not OVERLAY_SAMPLES_ON_OVERLAY:
    ax4.contourf(Xh, Yh, Zh, levels=[thr95_h, float(Zh.max())], colors=["#fdae6b"], alpha=0.28)
ax4.contour (Xh, Yh, Zh, levels=[thr95_h], colors=["#e6550d"], linewidths=[2.0], zorder=3)
ax4.contour (Xh, Yh, Zh, levels=[thr80_h], colors=["#e6550d"], linewidths=[1.4], linestyles="--", zorder=3)
ax4.contour (Xh, Yh, Zh, levels=[thr50_h], colors=["#e6550d"], linewidths=[1.2], linestyles="--", zorder=3)

# Optionally overlay raw samples
if OVERLAY_SAMPLES_ON_OVERLAY:
    ax4.scatter(eta_monkey, kap_monkey, s=SAMPLES_POINT_SIZE, alpha=SAMPLES_ALPHA, color="#2c7fb8", edgecolors="none", zorder=2)
    ax4.scatter(eta_human,  kap_human,  s=SAMPLES_POINT_SIZE, alpha=SAMPLES_ALPHA, color="#e6550d", edgecolors="none", zorder=2)

ax4.set_xlabel("Capacity η"); ax4.set_ylabel("Control κ")
ax4.grid(False)

# Tight, robust limits
all_eta = np.concatenate([eta_monkey, eta_human]); all_kap = np.concatenate([kap_monkey, kap_human])
x_lo, x_hi = np.percentile(all_eta, [0.5, 99.5]); y_lo, y_hi = np.percentile(all_kap, [0.5, 99.5])
xr = float(x_hi - x_lo); yr = float(y_hi - y_lo)
ax4.set_xlim(float(y_lo - 0.10 * yr), float(x_hi + 0.03 * xr))
ax4.set_ylim(float(y_lo - 0.10 * yr), float(y_hi + 0.10 * yr))

import matplotlib.colors as mcolors
legend_elems = [
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#fdae6b", 0.35),
        edgecolor=mcolors.to_rgba("#e6550d", 0.5),
        linewidth=1.0,
        label="Humans",
    ),
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#6baed6", 0.35),
        edgecolor=mcolors.to_rgba("#2c7fb8", 0.5),
        linewidth=1.0,
        label="Rhesus Macaques",
    ),
]
ax4.legend(
    handles=legend_elems,
    loc="upper left",
    frameon=False,
    handletextpad=0.12,
    labelspacing=0.15,
    handlelength=0.8,
    handleheight=0.45,
)

fig4.tight_layout()
out_overlay_2d_raw_pdf = os.path.join(OUT_DIR, f"{PREFIX}_2d_overlay_raw_{plot_type}.pdf")
fig4.savefig(out_overlay_2d_raw_pdf, dpi=400, bbox_inches="tight")
print(f"Saved: {out_overlay_2d_raw_pdf} ({plot_type} plot)")
plt.show()

#%%
# Raw posterior samples scatter in log space: Control (log κ) vs Capacity (log η)
fig_sc_log, (axm_l, axh_l) = plt.subplots(1, 2, figsize=(3.54, 2.36), dpi=400, sharex=True, sharey=True)

# Rhesus Macaques (axis 0)
axm_l.scatter(log_eta_monkey, log_kap_monkey, s=6, alpha=0.35, color="#2c7fb8")
axm_l.set_title("Rhesus Macaques")
axm_l.set_xlabel("log Capacity (log η)")
axm_l.set_ylabel("log Control (log κ)")
axm_l.grid(False)

# Humans (axis 1)
axh_l.scatter(log_eta_human, log_kap_human, s=6, alpha=0.35, color="#e6550d")
axh_l.set_title("Humans")
axh_l.set_xlabel("log Capacity (log η)")
axh_l.grid(False)

# Limits based on robust percentiles in log space
all_log_eta = np.concatenate([log_eta_monkey, log_eta_human])
all_log_kap = np.concatenate([log_kap_monkey, log_kap_human])
lx_lo, lx_hi = np.percentile(all_log_eta, [0.5, 99.5])
ly_lo, ly_hi = np.percentile(all_log_kap, [0.5, 99.5])
lxr = float(lx_hi - lx_lo); lyr = float(ly_hi - ly_lo)
axm_l.set_xlim(float(lx_lo - 0.10 * lxr), float(lx_hi + 0.03 * lxr))
axm_l.set_ylim(float(ly_lo - 0.10 * lyr), float(ly_hi + 0.10 * lyr))

fig_sc_log.tight_layout()
out_scatter_log = os.path.join(OUT_DIR, f"{PREFIX}_raw_samples_scatter_log.pdf")
fig_sc_log.savefig(out_scatter_log, dpi=400, bbox_inches="tight")
print(f"Saved: {out_scatter_log}")
plt.show()

#%%
# Raw posterior samples scatter: Control (κ) vs Capacity (η) - MEAN CORRECTED
fig_sc, (axm, axh) = plt.subplots(1, 2, figsize=(3.54, 2.36), dpi=400, sharex=True, sharey=True)

# Rhesus Macaques (axis 0)
axm.scatter(eta_monkey, kap_monkey, s=6, alpha=0.35, color="#2c7fb8")
axm.set_title("Rhesus Macaques")
axm.set_xlabel("Capacity (η)")
axm.set_ylabel("Control (κ)")
axm.grid(False)

# Humans (axis 1)
axh.scatter(eta_human, kap_human, s=6, alpha=0.35, color="#e6550d")
axh.set_title("Humans")
axh.set_xlabel("Capacity (η)")
axh.grid(False)

# Match limits to contour plot for consistency
axm.set_xlim(float(y_lo - 0.10 * yr), float(x_hi + 0.03 * xr))
axm.set_ylim(float(y_lo - 0.10 * yr), float(y_hi + 0.10 * yr))

fig_sc.tight_layout()
out_scatter = os.path.join(OUT_DIR, f"{PREFIX}_raw_samples_scatter.pdf")
fig_sc.savefig(out_scatter, dpi=400, bbox_inches="tight")
print(f"Saved: {out_scatter}")
plt.show()

#%%
# *** NEW: Proportion control using Monte Carlo means ***
# Single plot with overlaid species densities only
fig_pc, ax = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi=400)
ax.hist(pctrl_mon_mc, bins=60, range=(0, 1), density=True, alpha=0.6, color="#6baed6", label="Rhesus Macaques")
ax.hist(pctrl_hum_mc, bins=60, range=(0, 1), density=True, alpha=0.6, color="#fdae6b", label="Humans")
ax.set_ylabel("Density")
ax.set_xlabel("Proportion Control (" + r"$\frac{\kappa}{\kappa + \eta}$" + ")")
ax.set_xlim(0, 1)
ax.grid(False)

# Add legend for species info only
legend_elems = [
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#fdae6b", 0.35),
        edgecolor=mcolors.to_rgba("#e6550d", 0.5),
        linewidth=1.0,
        label="Humans",
    ),
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#6baed6", 0.35),
        edgecolor=mcolors.to_rgba("#2c7fb8", 0.5),
        linewidth=1.0,
        label="Rhesus Macaques",
    ),
]

ax.legend(
    handles=legend_elems,
    loc="upper right",
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.3,
    handlelength=0.8,
    handleheight=0.6,
    fontsize=10,
    ncol=1,
)

fig_pc.tight_layout()

# Save to out_dir folder
out_pctrl_pdf = os.path.join(OUT_DIR, f"{PREFIX}_pctrl.pdf")
fig_pc.savefig(out_pctrl_pdf, dpi=400, bbox_inches="tight")
print(f"Saved: {out_pctrl_pdf}")
plt.show()

#%%
# IPC plot (single panel with overlay only) - styled to match proportion control plot exactly
fig_ipc, ax = plt.subplots(1, 1, figsize=(3.54, 2.36), dpi=400)
ax.hist(ipc_mon, bins=60, density=True, alpha=0.6, color="#6baed6", label="Rhesus Macaques")
ax.hist(ipc_hum, bins=60, density=True, alpha=0.6, color="#fdae6b", label="Humans")
ax.set_ylabel("Density")
ax.set_xlabel("Information-Processing Capability (Bits)")
ax.grid(False)

# Add legend for species info only - same as proportion control
legend_ipc_elems = [
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#fdae6b", 0.35),
        edgecolor=mcolors.to_rgba("#e6550d", 0.5),
        linewidth=1.0,
        label="Humans",
    ),
    mpatches.Patch(
        facecolor=mcolors.to_rgba("#6baed6", 0.35),
        edgecolor=mcolors.to_rgba("#2c7fb8", 0.5),
        linewidth=1.0,
        label="Rhesus Macaques",
    ),
]

ax.legend(
    handles=legend_ipc_elems,
    loc="upper left",
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.3,
    handlelength=0.8,
    handleheight=0.6,
    fontsize=10,
    ncol=1,
)

ax.set_xlim(0, 1.58)

fig_ipc.tight_layout()

# Save to out_dir folder
out_ipc = os.path.join(OUT_DIR, f"{PREFIX}_ipc.pdf")
fig_ipc.savefig(out_ipc, dpi=400, bbox_inches="tight")
print(f"Saved: {out_ipc}")
plt.show()




#%%
# Sampling diagnostics from CmdStan CSVs (R-hat, ESS, divergences, treedepth, E-BFMI)
csv_dir = os.path.dirname(NPZ_FILE)
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
csv_files = [f for f in csv_files if os.path.isfile(f)]

if len(csv_files) == 0:
    print(f"No CSV files found under {csv_dir}; skipping diagnostics.")
else:
    import pandas as pd
    import numpy as np
    chains = []
    accept_stats = []
    treedepths = []
    leapfrogs = []
    divergences = []
    energies = []

    # Parameters of interest for summary diagnostics (compute R-hat/ESS)
    param_cols = [
        "mu_monkey.1", "mu_monkey.2",
        "beta_human.1", "beta_human.2",
        "sigma_monkey.1", "sigma_monkey.2",
        "sigma_human.1", "sigma_human.2",
        "L_Rho.1.1", "L_Rho.1.2", "L_Rho.2.1", "L_Rho.2.2",
        "nu"
    ]
    param_arrays = {c: [] for c in param_cols}

    for path in csv_files:
        df = pd.read_csv(path, comment="#")
        if df.empty:
            continue
        chains.append(path)
        accept_stats.append(float(df["accept_stat__"].mean()))
        treedepths.append({
            "mean": float(df["treedepth__"].mean()),
            "max": int(df["treedepth__"].max()),
            "saturations": int((df["treedepth__"] == df["treedepth__"].max()).sum()),
        })
        leapfrogs.append(float(df["n_leapfrog__"].mean()))
        divergences.append(int(df["divergent__"].sum()))
        energies.append(df["energy__"].to_numpy())
        # capture parameters
        for c in list(param_cols):
            if c in df.columns:
                param_arrays[c].append(df[c].to_numpy())
            else:
                # if missing (e.g., "nu" not present), also check alternative name "nu_log"
                if c == "nu" and "nu_log" in df.columns:
                    param_arrays[c].append(np.exp(df["nu_log"].to_numpy()))

    # E-BFMI per chain
    def e_bfmi(energy: np.ndarray) -> float:
        if energy.size < 2:
            return float("nan")
        diff = np.diff(energy)
        num = np.mean(diff * diff)
        den = np.var(energy)
        return float(num / den) if den > 0 else float("nan")

    ebfmis = [e_bfmi(e) for e in energies]

    # R-hat (split) implementation for selected parameters
    def split_rhat(chains_list: list[np.ndarray]) -> float:
        # drop to common length
        n = min(arr.size for arr in chains_list)
        if n < 10 or len(chains_list) < 2:
            return float("nan")
        arrs = [a[:n] for a in chains_list]
        # split
        split = []
        for a in arrs:
            half = n // 2
            split.append(a[:half])
            split.append(a[-half:])
        m = len(split)
        n2 = split[0].size
        chain_means = np.array([np.mean(s) for s in split])
        chain_vars = np.array([np.var(s, ddof=1) for s in split])
        B = n2 * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        var_hat = ((n2 - 1) / n2) * W + B / n2
        rhat = np.sqrt(var_hat / W) if W > 0 else float("nan")
        return float(rhat)

    # Simple ESS implementation without ArviZ dependency
    def ess_basic(chains_list: list[np.ndarray]) -> float:
        """Basic ESS computation using autocorrelation"""
        n = min(arr.size for arr in chains_list)
        if n < 10 or len(chains_list) < 2:
            return float("nan")
        
        # Concatenate all chains
        combined = np.concatenate([arr[:n] for arr in chains_list])
        
        # Compute autocorrelation up to lag n//4
        max_lag = min(n // 4, 100)
        autocorr = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr.append(1.0)
            else:
                # Pearson correlation with lag
                x1 = combined[:-lag]
                x2 = combined[lag:]
                if len(x1) > 0:
                    corr = np.corrcoef(x1, x2)[0, 1]
                    if np.isnan(corr):
                        break
                    autocorr.append(corr)
                else:
                    break
        
        # Find first negative autocorrelation or stop at max_lag
        cutoff = len(autocorr)
        for i, ac in enumerate(autocorr):
            if ac <= 0:
                cutoff = i
                break
        
        if cutoff <= 1:
            return float("nan")
        
        # ESS = N / (1 + 2 * sum of autocorrelations)
        ess = len(combined) / (1 + 2 * sum(autocorr[1:cutoff]))
        return float(ess)
    
    def ess_bulk(chains_list: list[np.ndarray]) -> float:
        return ess_basic(chains_list)
    
    def ess_tail(chains_list: list[np.ndarray]) -> float:
        return ess_basic(chains_list)

    diag_rows = []
    for p, lst in param_arrays.items():
        if len(lst) >= 2 and all(len(a) > 0 for a in lst):
            rh = split_rhat(lst)
            eb = ess_bulk(lst)
            et = ess_tail(lst)
            diag_rows.append({"param": p, "rhat": rh, "ess_bulk": eb, "ess_tail": et})

    # Aggregate
    total_div = int(sum(divergences))
    mean_accept = float(np.mean(accept_stats)) if accept_stats else float("nan")
    max_td = max([t["max"] for t in treedepths]) if treedepths else 0
    td_saturations = int(sum([t["saturations"] for t in treedepths]))
    mean_ebfmi = float(np.mean(ebfmis)) if ebfmis else float("nan")

    # Write report
    os.makedirs(OUT_DIR, exist_ok=True)
    report_txt = os.path.join(OUT_DIR, f"{PREFIX}_sampling_diagnostics.txt")
    with open(report_txt, "w") as f:
        f.write("Sampling diagnostics (CmdStan CSVs)\n")
        f.write(f"CSV dir: {csv_dir}\n")
        f.write(f"Chains: {len(chains)} files\n\n")
        f.write(f"Divergences (total): {total_div}\n")
        f.write(f"Mean accept_stat__: {mean_accept:.3f}\n")
        f.write(f"Max treedepth observed: {max_td} (saturations={td_saturations})\n")
        f.write(f"E-BFMI (mean across chains): {mean_ebfmi:.3f}\n\n")
        f.write("Per-parameter R-hat and ESS (selected parameters)\n")
        for row in diag_rows:
            f.write(f"  {row['param']}: Rhat={row['rhat']:.3f}, ESS_bulk={row['ess_bulk']:.1f}, ESS_tail={row['ess_tail']:.1f}\n")
    print(f"Saved diagnostics: {report_txt}")

    # Also CSV table
    report_csv = os.path.join(OUT_DIR, f"{PREFIX}_sampling_diagnostics.csv")
    pd.DataFrame(diag_rows).to_csv(report_csv, index=False)
    print(f"Saved diagnostics table: {report_csv}")

# %%
