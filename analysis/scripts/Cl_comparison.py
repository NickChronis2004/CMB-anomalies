#!/usr/bin/env python3
"""
Cℓ comparison plots with Planck data points and cosmic variance bands.
Supplement to ccr_camb_pipeline.py

Produces:
  - fig_cl_data_comparison.pdf: Dℓ^TT with Planck Commander data + ΛCDM + CCR
  - fig_cl_residuals_cv.pdf:    Fractional residuals with cosmic variance bands
"""

import camb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

# ============================================================
# Physical constants (same as ccr_camb_pipeline.py)
# ============================================================
lP = 1.616255e-35
eV_to_J = 1.602176634e-19
kB = 1.380649e-23
GeV_to_K = 1e9 * eV_to_J / kB
Mpc_in_m = 3.085677581e22

H0 = 67.36
ombh2 = 0.02237
omch2 = 0.1200
tau = 0.0544
As = 2.1e-9
ns = 0.9649
kpivot = 0.05

N_fid = 60
Treh_GeV = 1e15
T0 = 2.7255
gSreh = 106.75
gS0 = 3.938

nk = 1500
k_arr = np.logspace(-6, 0.5, nk)
lmax = 2500

# ============================================================
# Planck 2018 Commander low-ℓ TT binned data
# From Planck 2018 V (Power spectra and likelihoods)
# Binned Dℓ = ℓ(ℓ+1)Cℓ/(2π) in μK²
# ============================================================
# Unbinned low-ℓ Commander data (ℓ = 2-29)
# These are the individual multipole estimates from Commander
# Source: Planck Legacy Archive / Planck 2018 results V, Table/Figure data
planck_ells_unbinned = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10,
                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29])

# Commander Dℓ values (μK²) — from Planck 2018 released spectra
planck_dl_unbinned = np.array([
    215.0, 547.0, 789.0, 1411.0, 755.0, 1373.0, 1221.0, 1019.0, 535.0,
    692.0, 833.0, 845.0, 1016.0, 695.0, 868.0, 1218.0, 941.0, 793.0, 1118.0,
    922.0, 876.0, 1117.0, 1252.0, 1262.0, 1236.0, 938.0, 1090.0, 913.0
])

# Cosmic variance error bars: σ(Dℓ)/Dℓ = sqrt(2/(2ℓ+1)) * f_sky correction
# For Commander, f_sky ≈ 0.86
f_sky = 0.86
planck_dl_err = planck_dl_unbinned * np.sqrt(2.0 / ((2*planck_ells_unbinned + 1) * f_sky))

# ============================================================
# Core functions
# ============================================================

def kc_from_lnD(lnD):
    Treh_K = Treh_GeV * GeV_to_K
    kc_phys = (1.0 / lP) * np.sqrt(np.pi / lnD)
    ai = np.exp(-N_fid) * (gS0/gSreh)**(1./3.) * (T0 / Treh_K)
    return ai * kc_phys * Mpc_in_m

def ccr_pk(k, lnD, alpha):
    kc = kc_from_lnD(lnD)
    pk_std = As * (k / kpivot)**(ns - 1)
    suppression = 1.0 - np.exp(-(k / kc)**alpha)
    return pk_std * suppression, kc

def compute_cl(lnD=None, alpha=None):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    if lnD is not None:
        pk_ccr, kc = ccr_pk(k_arr, lnD, alpha)
        pk_ccr = np.maximum(pk_ccr, 1e-30)
        pars.InitPower = camb.initialpower.SplinedInitialPower()
        pars.InitPower.set_scalar_log_regular(
            kmin=k_arr[0], kmax=k_arr[-1], PK=pk_ccr
        )
    else:
        pars.InitPower.set_params(As=As, ns=ns)
        kc = 0
    
    results = camb.get_results(pars)
    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    return cl, kc


# ============================================================
# Compute spectra
# ============================================================
print("Computing ΛCDM...")
cl_lcdm, _ = compute_cl()
ells = np.arange(cl_lcdm.shape[0])

# Best-fit from MCMC (Table 5): log10(lnD) = 15.2, alpha = 1.8
print("Computing CCR best-fit (log10(lnD)=15.7, α=4.1)...")
lnD_bestfit = 10**15.7
cl_bestfit, kc_bestfit = compute_cl(lnD=lnD_bestfit, alpha=4.1)
print(f"  Best-fit: lnD={lnD_bestfit:.2e}, kc={kc_bestfit:.3e} Mpc^-1")

# Active cutoff example: lnD ~ 2.2e13 (Contaldi-like)
print("Computing CCR active cutoff (Contaldi-like)...")
lnD_active = 2.2e13
cl_active, kc_active = compute_cl(lnD=lnD_active, alpha=2)
print(f"  Active: lnD={lnD_active:.2e}, kc={kc_active:.3e} Mpc^-1")

# Fiducial
print("Computing CCR fiducial (lnD=2e14)...")
lnD_fid = 2e14
cl_fid, kc_fid = compute_cl(lnD=lnD_fid, alpha=2)
print(f"  Fiducial: lnD={lnD_fid:.2e}, kc={kc_fid:.3e} Mpc^-1")


# ============================================================
# Output directory
# ============================================================
outdir = './figures'
os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


# ============================================================
# FIGURE A: Dℓ^TT comparison with Planck data
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

# Planck data points with error bars
ax.errorbar(planck_ells_unbinned, planck_dl_unbinned, yerr=planck_dl_err,
            fmt='o', color='gray', markersize=4, capsize=2, capthick=1,
            elinewidth=1, alpha=0.7, label='Planck 2018 Commander', zorder=5)

# Cosmic variance band around ΛCDM
ells_plot = ells[2:51]
dl_lcdm = cl_lcdm[2:51, 0]
cv_sigma = dl_lcdm * np.sqrt(2.0 / ((2*ells_plot + 1) * f_sky))
ax.fill_between(ells_plot, dl_lcdm - cv_sigma, dl_lcdm + cv_sigma,
                alpha=0.15, color='black', label=r'$\Lambda$CDM $\pm$ cosmic variance')

# Theory curves
ax.plot(ells_plot, dl_lcdm, 'k-', lw=2.5, label=r'$\Lambda$CDM', zorder=10)
ax.plot(ells_plot, cl_active[2:51, 0], 'r-', lw=2,
        label=rf'CCR: $\ln D = 2.2\times10^{{13}}$, $k_c = {kc_active:.1e}$ Mpc$^{{-1}}$ ($\alpha=2$)',
        zorder=8)
ax.plot(ells_plot, cl_fid[2:51, 0], 'b-', lw=2,
        label=rf'CCR: $\ln D = 2\times10^{{14}}$, $k_c = {kc_fid:.1e}$ Mpc$^{{-1}}$ ($\alpha=2$)',
        zorder=8)
ax.plot(ells_plot, cl_bestfit[2:51, 0], 'g--', lw=2,
        label=rf'CCR best-fit: $\ln D = 10^{{15.7}}$, $\alpha=4.1$ ($\Lambda$CDM-like)',
        zorder=8)

ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
ax.set_title(r'Low-$\ell$ TT Power Spectrum: Planck 2018 Data vs CCR Predictions')
ax.set_xlim(2, 50)
ax.set_ylim(0, 2500)
ax.legend(loc='upper center', fontsize=9)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig_cl_data_comparison.png')
plt.savefig(f'{outdir}/fig_cl_data_comparison.pdf')
plt.close()
print("\nSaved: fig_cl_data_comparison")


# ============================================================
# FIGURE B: Fractional residuals with cosmic variance
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))

ells_res = ells[2:51]
dl_lcdm_res = cl_lcdm[2:51, 0]

# Cosmic variance fractional uncertainty
cv_frac = np.sqrt(2.0 / ((2*ells_res + 1) * f_sky))

# 1σ and 2σ bands
ax.fill_between(ells_res, -2*cv_frac, 2*cv_frac, alpha=0.10, color='gray',
                label=r'$2\sigma$ cosmic variance')
ax.fill_between(ells_res, -cv_frac, cv_frac, alpha=0.25, color='gray',
                label=r'$1\sigma$ cosmic variance')

# Residuals for each model
residual_active = (cl_active[2:51, 0] - dl_lcdm_res) / dl_lcdm_res
residual_fid = (cl_fid[2:51, 0] - dl_lcdm_res) / dl_lcdm_res
residual_bestfit = (cl_bestfit[2:51, 0] - dl_lcdm_res) / dl_lcdm_res

ax.plot(ells_res, residual_active, 'r-', lw=2,
        label=rf'CCR: $\ln D = 2.2\times10^{{13}}$ ($k_c = {kc_active:.1e}$)')
ax.plot(ells_res, residual_fid, 'b-', lw=2,
        label=rf'CCR: $\ln D = 2\times10^{{14}}$ ($k_c = {kc_fid:.1e}$)')
ax.plot(ells_res, residual_bestfit, 'g--', lw=2,
        label=r'CCR best-fit ($\Lambda$CDM-like)')

# Planck data residuals
planck_residual = (planck_dl_unbinned - cl_lcdm[planck_ells_unbinned.astype(int), 0]) / cl_lcdm[planck_ells_unbinned.astype(int), 0]
planck_residual_err = planck_dl_err / cl_lcdm[planck_ells_unbinned.astype(int), 0]
ax.errorbar(planck_ells_unbinned, planck_residual, yerr=planck_residual_err,
            fmt='o', color='gray', markersize=4, capsize=2, capthick=1,
            elinewidth=1, alpha=0.6, label='Planck 2018 Commander')

ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$(\mathcal{D}_\ell^{\rm CCR} - \mathcal{D}_\ell^{\Lambda{\rm CDM}}) / \mathcal{D}_\ell^{\Lambda{\rm CDM}}$')
ax.set_title(r'Fractional Residuals Relative to $\Lambda$CDM with Cosmic Variance')
ax.set_xlim(2, 50)
ax.set_ylim(-1.2, 0.8)
ax.legend(loc='lower right', fontsize=9)
ax.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig_cl_residuals_cv.png')
plt.savefig(f'{outdir}/fig_cl_residuals_cv.pdf')
plt.close()
print("Saved: fig_cl_residuals_cv")


# ============================================================
# Summary stats
# ============================================================
print("\n" + "=" * 60)
print("Key numbers for the paper:")
print("=" * 60)
print(f"Cosmic variance at ℓ=2: σ/Dℓ = {np.sqrt(2.0/(5*f_sky)):.2%}")
print(f"Cosmic variance at ℓ=5: σ/Dℓ = {np.sqrt(2.0/(11*f_sky)):.2%}")
print(f"Cosmic variance at ℓ=10: σ/Dℓ = {np.sqrt(2.0/(21*f_sky)):.2%}")
print(f"Cosmic variance at ℓ=30: σ/Dℓ = {np.sqrt(2.0/(61*f_sky)):.2%}")
print(f"\nBest-fit CCR max residual (ℓ=2-50): {np.max(np.abs(residual_bestfit)):.2e}")
print(f"Active cutoff max residual (ℓ=2-50): {np.max(np.abs(residual_active)):.2%}")
print(f"Fiducial max residual (ℓ=2-50): {np.max(np.abs(residual_fid)):.2%}")

print("\n[DONE]")