# CCR-CMB MCMC Pipeline
## Paper A: CCR-Motivated IR Cutoff and CMB Power Spectrum Suppression

### Files

| File | Description |
|------|-------------|
| `run_ccr_mcmc.py` | **Master script** — run with `--validate`, `--install-data`, `--run`, `--analyse` |
| `ccr_theory_cobaya.py` | Custom Cobaya Theory class (CAMB + CCR P(k)) |
| `ccr_cobaya.yaml` | Cobaya YAML config (alternative to Python script) |
| `ccr_camb_pipeline.py` | Standalone CAMB pipeline (validation + plots, no MCMC) |

### Quick Start

```bash
# 1. Install dependencies
pip install camb cobaya getdist matplotlib numpy scipy

# 2. Validate pipeline (no Planck data needed)
python run_ccr_mcmc.py --validate

# 3. Download Planck 2018 data (~1 GB)
python run_ccr_mcmc.py --install-data
# OR manually:
cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE \
    planck_2018_highl_plik.TTTEEE_lite -p ./packages
export COBAYA_PACKAGES_PATH=./packages

# 4. Run CCR MCMC
python run_ccr_mcmc.py --run

# 5. Run LCDM baseline (for Bayes factor)
python run_ccr_mcmc.py --run-lcdm

# 6. Analyse chains & make plots
python run_ccr_mcmc.py --analyse
```

### Parameter Space

| Parameter | Prior | Description |
|-----------|-------|-------------|
| `log_lnD` | Uniform [12, 16] | log₁₀(ln D) — CCR Hilbert space dimension |
| `alpha_ccr` | Uniform [1, 6] | Shape parameter of suppression |
| `H0` | Uniform [40, 100] | Hubble constant |
| `ombh2` | Uniform [0.005, 0.1] | Baryon density |
| `omch2` | Uniform [0.001, 0.99] | Cold dark matter density |
| `tau` | Gaussian(0.0544, 0.0073) | Optical depth |
| `As` | Uniform [5e-10, 5e-9] | Scalar amplitude |
| `ns` | Uniform [0.8, 1.2] | Spectral index |

### Derived Parameters

| Parameter | Equation |
|-----------|----------|
| `kc` | k_c(D) from eq. (11) |
| `lnD` | 10^(log_lnD) |
| `Hinf_GeV` | (1/l_P) √(π/ln D) × ℏc |

### Likelihoods

- **planck_2018_lowl.TT**: Commander, ℓ = 2–29 (most sensitive to CCR)
- **planck_2018_lowl.EE**: SimAll, constrains τ
- **planck_2018_highl_plik.TTTEEE_lite**: Plik lite, ℓ = 30–2508

### Convergence

- Target: Gelman-Rubin R-1 < 0.01
- Estimated runtime: ~18 hours with 4 parallel chains
- Each likelihood evaluation: ~5 seconds

### Thermal History (Fixed)

- N = 60 e-folds
- T_reh = 10¹⁵ GeV
- g*S,reh = 106.75, g*S,0 = 3.938

### Notes

- The `D → ∞` limit recovers ΛCDM to ~5×10⁻⁵ precision (spline interpolation limit)
- For proper Bayesian evidence computation, use `polychord` sampler instead of `mcmc`
- CAMB version validated: 1.6.5
- Cobaya version: 3.6.1
