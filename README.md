## Retrocue Experiment Analysis


### Data

For all data, each row is a subject/trial-type and contains information about that trial-type, number of trials and number correct responses.

Prefiltered data (includes all subjects):
- `data/human_agg_long_FullRun.csv`: human
- `data/monkey_agg_long_FullRun.csv`: macaque - Data from Brady and Hampton, 2018, Cognition

Filtered data (post removing 14 human subjects based on c and d'; no monkeys removed )
- `data/human_agg_long_FullRun_filtered.csv`: human, )
- `data/monkey_agg_long_FullRun_filtered.csv`: macaque - Data from Brady and Hampton, 2018, Cognition
- `data/stan_input_long_FullRun_filtered.csv`: combined human+macaque long-format for Stan

### Code

Basic data preprocessing:

- `qc_screen_subjects.py`
  - Human/monkey - flag bad subjects (computes d′ and c and flags subjects below thresholds). 
- `plot_qc_and_filter_data.py`
  - Summarizes signal detection theory flags used to remove subjects and writes filtered long-format datasets.

Training an emulator model to use for stan fits:
- `model_simulator.jl`
  - Julia simulator of recall model, used by the grid generator.
- `generate_emulator_grid_params.jl`
  - Julia script to generate (η, κ, ν, m, a, q) grid via LHS and call the simulator; used to create training data for the emulator. Note that this needs to be run via a parallel array job
- `train_emulator.py`
  - Trains the scikit-learn MLP emulator from simulator-generated output; exports .joblib (for pics) and .json (for stan)

Fitting stan model:
- `capacity_control_memory_robust.stan`
  -  Stan model for bayesian inference of capacity and control
- `fit_stan_model.py`
  - Prepares Stan data (using the combined long CSV) and fits model via CmdStanPy.

Plotting and analysis of stan outputs, comparison to data:
- `plot_ppc_data_posteriors_derived.py`
  - Loads posterior draws and emulator; plots posteriors over capacity and control, computes derived quantities (IPC, proportion control) and plots those as well.
  - Posterior predictive  plots and data-model comparison plots (accuracy vs load, pre-cue timing, and cue reliability).
- `run_GLMMs.ipynb`
  - Notebook for generalized linear mixed-effects models analysis of data


