# CCR-Motivated Infrared Cutoff and CMB Power Spectrum Suppression

Analysis code for constraining the Conditional Cosmological Recurrence (CCR) framework against Planck 2018 CMB data.

**Paper:** N. Chronis & N. Sifakis, *CCR-Motivated Infrared Cutoff and CMB Power Spectrum Suppression* (2026)

**Related publication:** N. Chronis & N. Sifakis, *Conditional Cosmological Recurrence in Finite Hilbert Spaces and Holographic Bounds Within Causal Patches*, Universe 12(1) (2026) 10, [doi:10.3390/universe12010010](https://doi.org/10.3390/universe12010010)

---

## Overview

The CCR framework predicts an infrared cutoff in the primordial power spectrum from the finite Hilbert space dimension *D* of the de Sitter causal patch during inflation. The cutoff is not a free parameter — it is a deterministic function of *D* alone, fixed by the Gibbons–Hawking entropy bound and standard thermal-history inputs. This repository contains the full analysis pipeline: from the modified primordial spectrum implementation through MCMC sampling to Fisher forecasts and model comparison.

### Key Equation

The comoving IR cutoff predicted by the CCR framework:

```text
kc = (e^{-N} / lP) × (g*S,0 / g*S,reh)^{1/3} × (T0 / Treh) × √(π / ln D)
```

The modified primordial power spectrum:

```text
P_CCR(k) = As (k/k*)^{ns-1} [1 - exp(-(k/kc)^α)]
```

For fiducial parameters (N = 60, Treh = 10¹⁵ GeV, ln D = 2×10¹⁴): **kc ≈ 1.6 × 10⁻⁴ Mpc⁻¹**.

---

## Project Structure

```text
CMB-anomalies - JCAP/
├── analysis/
│   ├── scripts/                      # All analysis scripts
│   │   ├── ccr_theory_cobaya.py      # CCR primordial spectrum provider for Cobaya
│   │   ├── ccr_camb_pipeline.py      # CAMB integration and spectrum computation
│   │   ├── ccr_cobaya.yaml           # Cobaya MCMC configuration
│   │   ├── run_ccr_mcmc.py           # Main MCMC runner script
│   │   ├── analyse_ccr_chains.py     # Chain analysis and posterior plotting
│   │   ├── bayes_factor_analysis.py  # Savage-Dickey, AIC, BIC model comparison
│   │   ├── sensitivity_analysis.py   # Parameter sensitivity and contour plots
│   │   ├── Prior_sensetivity.py      # Prior sensitivity / importance reweighting
│   │   ├── Fisher_forecast.py        # Fisher forecast for future experiments
│   │   ├── Cl_comparison.py          # C_ℓ comparison with Planck data
│   │   ├── consistency_plot.py       # (kc, r) consistency relation plot
│   │   ├── tensor_ratio_derivation.py# Tensor-to-scalar ratio r(D) prediction
│   │   ├── appendix_convergence.py   # MCMC convergence diagnostics (Appendix B)
│   │   ├── appendix_validation.py    # Numerical validation tests (Appendix C)
│   │   ├── planck/                   # Planck likelihood data and code
│   │   └── README_mcmc.md            # MCMC-specific documentation
│   ├── chains/                       # MCMC output chains
│   │   ├── ccr_mcmc.1.txt            # Chain samples
│   │   ├── ccr_mcmc.covmat           # Proposal covariance matrix
│   │   ├── ccr_mcmc.checkpoint       # Sampler checkpoint
│   │   └── ccr_mcmc.progress         # Convergence progress
│   └── tables/                       # LaTeX tables for paper
│
├── paper/                            # LaTeX manuscript source
│   ├── figures/                      # All paper figures (PDF and PNG)
│   ├── main.tex                      # Main LaTeX file
│   ├── section2.tex … section5.tex   # Section source files
│   └── appendicies.tex               # Appendices source
│
├── requirements.txt                  # Python dependencies
├── LICENSE                           # GNU GPL v3.0
├── .gitignore
└── README.md                         # This file
```

### Figures

All figures (PDF and PNG) live in **`paper/figures/`**, which is the directory referenced by `\graphicspath{{figures/}}` in the LaTeX manuscript. Analysis scripts write their output directly to this directory.

---

## Requirements

- **Python** 3.14.0
- **OS:** Windows 10/11 (developed and tested)

### Python Packages

| Package    | Version | Description                              |
|------------|---------|------------------------------------------|
| cobaya     | 3.6.1   | MCMC sampler and likelihood framework    |
| camb       | 1.6.5   | Boltzmann solver for CMB power spectra   |
| numpy      | 2.2.6   | Numerical computing                      |
| scipy      | 1.16.3  | Scientific computing                     |
| matplotlib | 3.10.7  | Plotting                                 |
| getdist    | 1.7.5   | MCMC chain analysis and triangle plots   |

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Planck 2018 Data

The analysis requires Planck 2018 likelihood data, which must be downloaded separately:

1. Download the Planck 2018 baseline likelihoods from the [Planck Legacy Archive](https://pla.esac.esa.int/)
2. Place them in `scripts/packages/data/planck_2018/baseline/plc_3.0/`

Required likelihoods:

- **Low-ℓ TT:** Commander (`commander_dx12_v3_2_29.clik`)
- **Low-ℓ EE:** SimAll (`simall_100x143_offlike5_EE_Aplanck_B.clik`)
- **High-ℓ TT+TE+EE:** Plik-lite (`plik_lite_v22_TTTEEE.clik`)

---

## How to Run

All scripts are run from the `scripts/` directory:

```bash
cd scripts
```

### 1. MCMC Analysis

```bash
python run_ccr_mcmc.py
```

Runs the full 8-parameter MCMC (6 ΛCDM + 2 CCR: log₁₀(ln D) and α) using Cobaya with Metropolis-Hastings. Configuration is in `ccr_cobaya.yaml`.

**Expected runtime:** ~8–12 hours on a modern CPU (convergence target R−1 < 0.01).

### 2. Chain Analysis and Posteriors

```bash
python analyse_ccr_chains.py
```

Generates posterior distributions, triangle plots, and parameter constraints (Table 6, Figures 5–6 in the paper).

### 3. Model Comparison

```bash
python bayes_factor_analysis.py
```

Computes the Savage-Dickey density ratio (ln B₀₁ = 0.06), AIC (+4.0), and BIC (+13.0) for CCR vs ΛCDM.

### 4. Theory and Sensitivity

```bash
python sensitivity_analysis.py
```

Produces parameter sensitivity tables and contour plots for kc in the (N, ln D) and (Treh, ln D) planes (Table 2, Figure 2).

### 5. Prior Sensitivity Analysis

```bash
python Prior_sensitivity.py
```

Importance reweighting under alternative prior configurations (Table 7, Figure 9).

### 6. Fisher Forecast

```bash
python Fisher_forecast.py
```

Detection significance forecasts for Planck, LiteBIRD, and CMB-S4 (Tables 9, Figures 11–12).

### 7. Cℓ Comparison with Planck Data

```bash
python Cl_comparison.py
```

Low-ℓ TT power spectrum comparison and fractional residuals (Figures 7–8).

### 8. Consistency Plot (kc vs r)

```bash
python consistency_plot.py
```

CCR consistency relation in the (kc, r) plane (Figure 10). Saves to `../figures/`.

### 9. Tensor-to-Scalar Ratio

```bash
python tensor_ratio_derivation.py
```

CCR prediction for r as a function of ln D (Figure 4, Table 4).

### 10. Appendix Diagnostics

```bash
python appendix_convergence.py    # Trace plots, Gelman-Rubin (Figures 13–15)
python appendix_validation.py     # ΛCDM recovery and k-grid tests (Figure 16)
```

---

## Key Parameters

| Parameter       | Prior           | Description                   |
|-----------------|-----------------|-------------------------------|
| log10(ln D)     | U[12, 16]       | Hilbert space dimension       |
| alpha           | U[1, 6]         | Suppression shape parameter   |
| Omega_b h^2     | U[0.005, 0.1]   | Baryon density                |
| Omega_c h^2     | U[0.001, 0.99]  | CDM density                   |
| H0              | U[40, 100]      | Hubble constant (km/s/Mpc)    |
| tau             | N(0.054, 0.007) | Optical depth                 |
| As              | U[5e-10, 5e-9]  | Scalar amplitude              |
| ns              | U[0.8, 1.2]     | Spectral index                |

Fixed parameters: Σmν = 0.06 eV, Ωk = 0.

---

## Principal Results

- The Planck data are **consistent** with the CCR framework but **do not prefer** it over ΛCDM.
- Savage-Dickey Bayes factor: **ln B₀₁ = 0.06** (no preference).
- Posterior on log₁₀(ln D) is approximately **uniform** over [12, 16].
- CCR parameters show **no correlations** with standard ΛCDM parameters.
- A **3σ detection** requires ln D ≲ 10¹³ (kc ≳ 7 × 10⁻⁴ Mpc⁻¹).
- The CCR framework simultaneously predicts **r = 16/(As ln D)**, providing a two-observable consistency test.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Chronis2026ccr_cmb,
  author  = {Chronis, N. and Sifakis, N.},
  title   = {CCR-Motivated Infrared Cutoff and CMB Power Spectrum Suppression},
  year    = {2026}
}

@article{Chronis2026ccr,
  author  = {Chronis, N. and Sifakis, N.},
  title   = {Conditional Cosmological Recurrence in Finite Hilbert Spaces
             and Holographic Bounds Within Causal Patches},
  journal = {Universe},
  volume  = {12},
  number  = {1},
  pages   = {10},
  year    = {2026},
  doi     = {10.3390/universe12010010}
}
```

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
