#%%
import pandas as pd
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

human_agg_long_FullRun = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "human_agg_long_FullRun.csv"))
monkey_agg_long_FullRun = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "monkey_agg_long_FullRun.csv"))

# %%

# Load QC results
human_qc = pd.read_csv(os.path.join(PROJECT_ROOT, "qc_subject_bias_screen_FullRun.csv"))
monkey_qc = pd.read_csv(os.path.join(PROJECT_ROOT, "qc_monkey_bias_screen_FullRun.csv"))

# Filter out flagged subjects
human_qc_flagged = human_qc[human_qc["exclude_any"] == True]
monkey_qc_flagged = monkey_qc[monkey_qc["exclude_any"] == True]

print("Human QC Results:")
print(f"  Total subjects: {len(human_qc)}")
print(f"  Flagged for |c| > 0.5: {human_qc['flag_abs_c_gt_thresh'].sum()}")
print(f"  Flagged for d' < 0.25: {human_qc['flag_dprime_lt_thresh'].sum()}")
print(f"  Total flagged (exclude_any): {human_qc['exclude_any'].sum()}")
print(f"  Remaining subjects: {len(human_qc) - human_qc['exclude_any'].sum()}")

print("\nMonkey QC Results:")
print(f"  Total monkeys: {len(monkey_qc)}")
print(f"  Flagged for |c| > 0.5: {monkey_qc['flag_abs_c_gt_thresh'].sum()}")
print(f"  Flagged for d' < 0.25: {monkey_qc['flag_dprime_lt_thresh'].sum()}")
print(f"  Total flagged (exclude_any): {monkey_qc['exclude_any'].sum()}")
print(f"  Remaining monkeys: {len(monkey_qc) - monkey_qc['exclude_any'].sum()}")

# Create filtered datasets
human_agg_long_FullRun_filtered = human_agg_long_FullRun[
    ~human_agg_long_FullRun["id"].isin(human_qc_flagged["subject_id"])
].copy()

monkey_agg_long_FullRun_filtered = monkey_agg_long_FullRun[
    ~monkey_agg_long_FullRun["id"].isin(monkey_qc_flagged["monkey_id"])
].copy()

print(f"\nFiltered datasets:")
print(f"  Human rows: {len(human_agg_long_FullRun)} → {len(human_agg_long_FullRun_filtered)}")
print(f"  Monkey rows: {len(monkey_agg_long_FullRun)} → {len(monkey_agg_long_FullRun_filtered)}")
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global typography settings to match plot_stan_posteriors_ma_interactive.py
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'STIXGeneral', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
import numpy as np


# Create histograms for humans - PNAS 2-column width style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5), dpi = 300)  # PNAS 2-column width

# d' histogram
ax1.hist(human_qc['d_prime'], bins=30, alpha=0.7, edgecolor='black', color='#6baed6')
ax1.axvline(x=0.25, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Sensitivity (d\')')  # APA standard
ax1.set_ylabel('Frequency')  # APA standard
ax1.set_title('Human Sensitivity Distribution')
ax1.grid(False)  # Remove grid to match other plots

# criterion_c histogram
ax2.hist(human_qc['criterion_c'], bins=30, alpha=0.7, edgecolor='black', color='#6baed6')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=-0.5, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Criterion Location (c)')  # APA standard
ax2.set_ylabel('Frequency')
ax2.set_title('Human Criterion Distribution')
ax2.grid(False)

plt.tight_layout()
plt.show()

# Create histograms for monkeys - PNAS 2-column width style  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5), dpi = 300)  # PNAS 2-column width

# d' histogram
ax1.hist(monkey_qc['d_prime'], bins=20, alpha=0.7, edgecolor='black', color='#fdae6b')
ax1.axvline(x=0.25, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Sensitivity (d\')')  # APA standard
ax1.set_ylabel('Frequency')  # APA standard  
ax1.set_title('Rhesus Macaque Sensitivity Distribution')
ax1.grid(False)  # Remove grid to match other plots

# criterion_c histogram
ax2.hist(monkey_qc['criterion_c'], bins=20, alpha=0.7, edgecolor='black', color='#fdae6b')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=-0.5, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Criterion Location (c)')  # APA standard
ax2.set_ylabel('Frequency')
ax2.set_title('Rhesus Macaque Criterion Distribution')
ax2.grid(False)

plt.tight_layout()
plt.show()

# Save filtered datasets
human_agg_long_FullRun_filtered.to_csv(os.path.join(PROJECT_ROOT, "data", "human_agg_long_FullRun_filtered.csv"), index=False)
monkey_agg_long_FullRun_filtered.to_csv(os.path.join(PROJECT_ROOT, "data", "monkey_agg_long_FullRun_filtered.csv"), index=False)

# Also create a combined filtered Stan input
combined_filtered = pd.concat([human_agg_long_FullRun_filtered, monkey_agg_long_FullRun_filtered], ignore_index=True)
stan_long_filtered = combined_filtered[combined_filtered["type"].isin([0,1])].copy().reset_index(drop=True)
stan_long_filtered = stan_long_filtered.sort_values(["species","id","q","m","pre_ms","type"]).reset_index(drop=True)
stan_long_filtered.to_csv(os.path.join(PROJECT_ROOT, "data", "stan_input_long_FullRun_filtered.csv"), index=False)

print(f"\nSaved filtered datasets:")
print(f"  data/human_agg_long_FullRun_filtered.csv ({len(human_agg_long_FullRun_filtered)} rows)")
print(f"  data/monkey_agg_long_FullRun_filtered.csv ({len(monkey_agg_long_FullRun_filtered)} rows)")
print(f"  data/stan_input_long_FullRun_filtered.csv ({len(stan_long_filtered)} rows)")