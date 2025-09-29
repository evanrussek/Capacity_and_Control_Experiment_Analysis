functions {
  real relu(real x) { return fmax(0.0, x); }
  real log10_(real x) { return log(x) / log(10.0); }

  // Emulator for 8-D inputs: [log10 eta, log10 kappa, log10 nu, q, a, m_2, m_3, m_4]
  vector emu_logits_2layer(
      real eta, real kappa, real nu,
      real q, real a,
      real m2, real m3, real m4,
      int D, int H1, int H2, int O,
      vector x_mean, vector x_scale,
      matrix W1, vector b1,
      matrix W2, vector b2,
      matrix W3, vector b3
  ) {
    vector[D] x;
    vector[D] z;
    vector[H1] h1;
    vector[H2] h2;
    vector[O] out;

    x[1] = log10_(eta);
    x[2] = log10_(kappa);
    x[3] = log10_(nu);
    x[4] = q;
    x[5] = a;
    x[6] = m2;
    x[7] = m3;
    x[8] = m4;

    for (d in 1:D) z[d] = (x[d] - x_mean[d]) / x_scale[d];
    h1 = to_vector(z' * W1) + b1;
    for (i in 1:H1) h1[i] = relu(h1[i]);
    h2 = to_vector(h1' * W2) + b2;
    for (i in 1:H2) h2[i] = relu(h2[i]);
    out = to_vector(h2' * W3) + b3; // [valid, invalid] logits
    return out;
  }

  real beta_binomial_mu_tau_lpmf(int k, int n, real mu, real tau) {
    real a = fmax(1e-6, mu * tau);
    real b = fmax(1e-6, (1 - mu) * tau);
    return lchoose(n, k) + lbeta(k + a, n - k + b) - lbeta(a, b);
  }
}

data {
  int<lower=1> K;
  int<lower=1> J;
  int<lower=1> S;
  array[K] int<lower=1,upper=J> pid;
  array[J] int<lower=1,upper=S> species_id;

  array[K] int<lower=0,upper=1> type;   // 1=valid, 0=invalid
  array[K] int<lower=0> y;
  array[K] int<lower=0> n;
  array[K] real<lower=0,upper=1> q;
  array[K] real m;                      // 2/3/4 (numeric)
  array[K] real<lower=0,upper=1> a;     // T_pre / 1350

  // Priors
  real mu_mean;
  real<lower=0> mu_sd;
  real<lower=0> sigma_rate;
  real<lower=1> lkj_eta;
  real<lower=0> beta_sd;

  // Emulator weights & shapes (D==8)
  int<lower=8> D;
  int<lower=1> H1;
  int<lower=1> H2;
  int<lower=2> O;
  vector[D] x_mean;
  vector[D] x_scale;
  matrix[D, H1] W1;
  vector[H1]    b1;
  matrix[H1, H2] W2;
  vector[H2]     b2;
  matrix[H2, O]  W3;
  vector[O]      b3;

  // Domain guards
  real<lower=0> eta_lo; real<lower=0> eta_hi;
  real<lower=0> kap_lo; real<lower=0> kap_hi;
  real<lower=0> nu_lo;  real<lower=0> nu_hi;

  // Optional species-level fixed τ (set to 0 to disable)
  vector<lower=0>[S] tau_obs;
}

parameters {
  // Hierarchy on (log eta, log kappa)
  vector[2] mu_monkey;
  vector[2] beta_human;
  vector<lower=0>[2] sigma_monkey;
  vector<lower=0>[2] sigma_human;
  cholesky_factor_corr[2] L_Rho;
  matrix[2, J] z_raw;

  real nu_log;

  // -- per-subject overdispersion (Beta–Binomial) ---
  vector[2] mu_tau_log;                  // species-level means (log τ)
  vector<lower=0>[2] sigma_tau_log;      // species-level scatters
  vector[J] z_tau;                       // subject deviations

  // -- per-subject lapse (toward chance 0.5) ---
  vector[J] epsilon_logit;               // logit(ε_j)
}

transformed parameters {
  matrix[2, J] log_etakappa;
  vector[K] p;           // emulator mean
  vector[K] p_obs;       // after lapse mixing
  real nu = exp(nu_log);
  // Clamp ν to emulator training domain
  real nu_emu = fmin(fmax(nu,  nu_lo),  nu_hi);

  // Subject-level τ and ε
  vector<lower=0>[J] tau_subj;
  vector<lower=0,upper=1>[J] epsilon;

  for (j in 1:J) {
    // build (log eta, log sqrt(kappa))
    vector[2] mu_j;
    vector[2] sd_j;
    if (species_id[j] == 2) {
      mu_j = mu_monkey + beta_human;
      sd_j = sigma_human;
    } else {
      mu_j = mu_monkey;
      sd_j = sigma_monkey;
    }
    log_etakappa[, j] = mu_j + diag_pre_multiply(sd_j, L_Rho) * z_raw[, j];

    // --- per-subject τ and ε
    tau_subj[j] = exp(mu_tau_log[species_id[j]] + sigma_tau_log[species_id[j]] * z_tau[j]);
    epsilon[j]  = inv_logit(epsilon_logit[j]);
  }

  // Emulator predictions and lapse mixture
  for (i in 1:K) {
    real eta   = exp(log_etakappa[1, pid[i]]);
    real kappa = exp(log_etakappa[2, pid[i]]);
    real m2 = (m[i] == 2) ? 1.0 : 0.0;
    real m3 = (m[i] == 3) ? 1.0 : 0.0;
    real m4 = (m[i] == 4) ? 1.0 : 0.0;

    // Clamp η, κ to emulator training domain
    real eta_emu = fmin(fmax(eta,   eta_lo), eta_hi);
    real kap_emu = fmin(fmax(kappa, kap_lo), kap_hi);

    vector[2] z = emu_logits_2layer(
      eta_emu, kap_emu, nu_emu, q[i], a[i], m2, m3, m4,
      D, H1, H2, O, x_mean, x_scale, W1, b1, W2, b2, W3, b3
    );
    real p_valid   = inv_logit(z[1]);
    real p_invalid = inv_logit(z[2]);
    p[i] = type[i] ? p_valid : p_invalid;

    // --- lapse toward chance (0.5)
    p_obs[i] = (1.0 - epsilon[pid[i]]) * p[i] + epsilon[pid[i]] * 0.5;
  }
}

model {
  // Priors for η/κ hierarchy
  mu_monkey  ~ normal(mu_mean, mu_sd);
  beta_human ~ normal(0, beta_sd);
  sigma_monkey ~ exponential(sigma_rate);
  sigma_human  ~ exponential(sigma_rate);
  L_Rho ~ lkj_corr_cholesky(lkj_eta);
  to_vector(z_raw) ~ std_normal();

  // Prior for ν
  nu_log ~ normal(log(sqrt(nu_lo * nu_hi)), 0.75);

  // --- priors for τ_subj hierarchy (log scale) ---
  // Center log τ around ~log(100): near-binomial but allows dispersion
  mu_tau_log     ~ normal(log(100), 1.0);
  sigma_tau_log  ~ exponential(1.0);
  z_tau          ~ std_normal();

  // --- prior for lapse ε_j ---
  // Center near small lapse (e.g., 2%) with some spread
  epsilon_logit  ~ normal(logit(0.02), 1.0);

  // Likelihood (Beta–Binomial with effective τ)
  for (i in 1:K) {
    int s = species_id[pid[i]];
    real tau_eff = (tau_obs[s] > 0) ? tau_obs[s] : tau_subj[pid[i]];
    target += beta_binomial_mu_tau_lpmf(y[i] | n[i], p_obs[i], tau_eff);
  }
}

generated quantities {
  vector[K] log_lik;
  for (i in 1:K) {
    int s = species_id[pid[i]];
    real tau_eff = (tau_obs[s] > 0) ? tau_obs[s] : tau_subj[pid[i]];
    log_lik[i] = beta_binomial_mu_tau_lpmf(y[i] | n[i], p_obs[i], tau_eff);
  }
}


