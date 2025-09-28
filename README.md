## Retrocue Experiment Analysis


### Folder layout

- `code/`: analysis scripts and model files
- `data/`: de-identified aggregated datasets

### Data

Filtered data (post removing 14 human subjects based on c and d' )
- `data/human_agg_long_FullRun_filtered.csv`: human, aggregated long-format (filtered based on input)
- `data/monkey_agg_long_FullRun_filtered.csv`: macaque, aggregated long-format (QC-filtered)
- `data/stan_input_long_FullRun_filtered.csv`: combined human+macaque long-format for Stan

Prefiltered data (includes all subjects - note monkey data is the same):
- `data/human_agg_long_FullRun.csv`: human, aggregated long-format (pre-filtered; before exclusions)
- `data/monkey_agg_long_FullRun.csv`: macaque, aggregated long-format (pre-filtered)
- `data/stan_input_long_FullRun.csv`: combined human+macaque long-format for Stan (pre-filtered)

### Scripts and models (code/)

- `capacity_control_memory_robust.stan`
  -  Stan model for bayesian inference of capacity and control
- `fit_stan_model.py`
  - Prepares Stan data (using the combined long CSV) and fits model via CmdStanPy.
- `plot_posteriors_and_derived.py`
  - Loads posterior draws and emulator; plots posteriors, computes derived quantities (e.g., IPC, proportion control) and plots those as well.
- `plot_ppc_and_data.py`
  - Posterior predictive check plots and data-model comparison plots (accuracy vs load, pre-cue timing, and cue reliability).
- `plot_qc_and_filter_data.py`
  - Summarizes signal detection theory flags used to remove subjects and writes filtered long-format datasets.
- `qc_screen_subjects.py`
  - Human/monkey - flag bad subjects (computes d′ and c and flags subjects below thresholds). 
- `train_emulator.py`
  - Trains the scikit-learn MLP emulator from simulator-generated output; exports `mlp_emulator.joblib` and `emulator_for_stan.json`.
- `generate_emulator_grid_params.jl`
  - Julia script to generate (η, κ, ν, m, a, q) grid via LHS and call the simulator; used to create training data for the emulator. Note that this needs to be run via a parallel array job
- `model_simulator.jl`
  - Julia simulator of recall model, used by the grid generator.
- `run_GLMMs.ipynb`
  - Notebook for generalized linear mixed-effects models analysis


