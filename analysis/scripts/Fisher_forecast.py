#!/usr/bin/env python3
"""
Fisher Forecast: LiteBIRD sensitivity to CCR parameters
========================================================
Estimates the improvement in constraining ln D from adding
low-ℓ EE and BB polarisation (LiteBIRD) to Planck TT.

Method: Simple Fisher matrix approach comparing signal-to-noise
of the CCR suppression across experiments.
"""

import camb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

# ============================================================
# Physical constants (same as pipeline)
# ============================================================
lP = 1.616255e-35
eV_to_J = 1.602176634e-19
kB = 1.380649e-23
GeV_to_K = 1e9 * eV_to_J / kB
Mpc_in_m = 3.085677581e22

H0 = 67.36; ombh2 = 0.02237; omch2 = 0.1200; tau = 0.0544
As = 2.1e-9; ns = 0.9649; kpivot = 0.05
N_fid = 60; Treh_GeV = 1e15; T0 = 2.7255; gSreh = 106.75; gS0 = 3.938
nk = 1500; k_arr = np.logspace(-6, 0.5, nk); lmax = 2500

# ============================================================
# Core functions
# ============================================================
def kc_from_lnD(lnD):
    Treh_K = Treh_GeV * GeV_to_K
    kc_phys = (1.0 / lP) * np.sqrt(np.pi / lnD)
    ai = np.exp(-N_fid) * (gS0/gSreh)**(1./3.) * (T0 / Treh_K)
    return ai * kc_phys * Mpc_in_m

def compute_cl(lnD=None, alpha=None):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    if lnD is not None:
        kc = kc_from_lnD(lnD)
        pk = As * (k_arr/kpivot)**(ns-1) * (1.0 - np.exp(-(k_arr/kc)**alpha))
        pk = np.maximum(pk, 1e-30)
        pars.InitPower = camb.initialpower.SplinedInitialPower()
        pars.InitPower.set_scalar_log_regular(kmin=k_arr[0], kmax=k_arr[-1], PK=pk)
    else:
        pars.InitPower.set_params(As=As, ns=ns)
    results = camb.get_results(pars)
    # Get TT, EE, BB, TE
    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    return cl

# ============================================================
# Experimental specifications
# ============================================================
# Noise: N_ℓ = (σ_noise)^2 * beam^2, but at low-ℓ beam is irrelevant
# We use effective noise per multipole

# Planck: TT only at low-ℓ (Commander), EE (SimAll)
# σ_T ~ 2 μK·arcmin (effective at low-ℓ, but cosmic variance dominated)
# σ_P ~ 55 μK·arcmin (HFI 143 GHz)
planck_fsky = 0.86
planck_noise_TT_muK2 = 0.0  # negligible vs cosmic variance at low-ℓ
planck_noise_EE_muK2 = (55.0 * np.pi/180/60)**2  # (μK·rad)^2

# LiteBIRD: full-sky polarisation
# σ_P ~ 2.2 μK·arcmin (combined), fsky ~ 0.70
litebird_fsky = 0.70
litebird_noise_EE_muK2 = (2.2 * np.pi/180/60)**2

# CMB-S4: ground-based, partial sky
# σ_P ~ 1.0 μK·arcmin, fsky ~ 0.40
cmbs4_fsky = 0.40
cmbs4_noise_EE_muK2 = (1.0 * np.pi/180/60)**2

# ============================================================
# Fisher matrix computation
# ============================================================
def fisher_snr(cl_lcdm, cl_ccr, fsky, noise_TT, noise_EE, ell_min=2, ell_max=30,
               use_TT=True, use_EE=True):
    """
    Compute cumulative signal-to-noise ratio for detecting the CCR
    suppression relative to ΛCDM.
    
    SNR^2 = sum_ℓ (2ℓ+1) f_sky sum_X [(Cℓ^CCR_X - Cℓ^LCDM_X)^2 / (Cℓ^LCDM_X + Nℓ_X)^2]
    
    This is the Fisher information for a single parameter (amplitude of suppression).
    """
    snr2 = 0.0
    snr2_cumulative = []
    ells_out = []
    
    for ell in range(ell_min, ell_max+1):
        term = 0.0
        
        if use_TT:
            # TT: index 0 in CAMB output (Dℓ = ℓ(ℓ+1)Cℓ/2π in μK²)
            # Convert Dℓ to Cℓ for proper noise addition
            dl_lcdm = cl_lcdm[ell, 0]
            dl_ccr = cl_ccr[ell, 0]
            cl_TT = dl_lcdm * 2 * np.pi / (ell*(ell+1))
            nl_TT = noise_TT  # already in Cℓ units
            
            delta_dl = dl_ccr - dl_lcdm
            # Variance of Dℓ: cosmic variance + noise
            var_dl = 2.0 / ((2*ell+1) * fsky) * (dl_lcdm + nl_TT * ell*(ell+1)/(2*np.pi))**2
            term += delta_dl**2 / var_dl
        
        if use_EE:
            dl_lcdm_ee = cl_lcdm[ell, 1]
            dl_ccr_ee = cl_ccr[ell, 1]
            nl_EE_dl = noise_EE * ell*(ell+1) / (2*np.pi)  # convert noise Cℓ to Dℓ
            
            delta_dl_ee = dl_ccr_ee - dl_lcdm_ee
            var_dl_ee = 2.0 / ((2*ell+1) * fsky) * (dl_lcdm_ee + nl_EE_dl)**2
            if var_dl_ee > 0:
                term += delta_dl_ee**2 / var_dl_ee
        
        snr2 += term
        snr2_cumulative.append(np.sqrt(snr2))
        ells_out.append(ell)
    
    return np.array(ells_out), np.array(snr2_cumulative), np.sqrt(snr2)


# ============================================================
# Compute spectra for a grid of ln D values
# ============================================================
print("Computing ΛCDM baseline...")
cl_lcdm = compute_cl()

# Test models: range of ln D with observable suppression
lnD_test_values = [5e12, 1e13, 2.2e13, 5e13, 1e14, 2e14]
alpha_fid = 2

results_table = []

print("\nComputing Fisher SNR for each experiment and ln D value...\n")
print(f"{'lnD':>12s} {'kc [Mpc^-1]':>14s} {'Planck TT':>12s} {'Planck TT+EE':>14s} "
      f"{'LiteBIRD EE':>13s} {'Planck+LiteBIRD':>16s} {'CMB-S4 EE':>11s}")
print("-" * 100)

for lnD in lnD_test_values:
    cl_ccr = compute_cl(lnD=lnD, alpha=alpha_fid)
    kc = kc_from_lnD(lnD)
    
    # Planck TT only
    _, _, snr_planck_TT = fisher_snr(cl_lcdm, cl_ccr, planck_fsky,
                                      planck_noise_TT_muK2, planck_noise_EE_muK2,
                                      use_TT=True, use_EE=False)
    
    # Planck TT+EE
    _, _, snr_planck_TTEE = fisher_snr(cl_lcdm, cl_ccr, planck_fsky,
                                        planck_noise_TT_muK2, planck_noise_EE_muK2,
                                        use_TT=True, use_EE=True)
    
    # LiteBIRD EE only
    _, _, snr_litebird = fisher_snr(cl_lcdm, cl_ccr, litebird_fsky,
                                     0, litebird_noise_EE_muK2,
                                     use_TT=False, use_EE=True)
    
    # Planck TT + LiteBIRD EE (combined)
    # Approximate: add Fisher informations
    snr_combined = np.sqrt(snr_planck_TT**2 + snr_litebird**2)
    
    # CMB-S4 EE only
    _, _, snr_cmbs4 = fisher_snr(cl_lcdm, cl_ccr, cmbs4_fsky,
                                  0, cmbs4_noise_EE_muK2,
                                  use_TT=False, use_EE=True)
    
    results_table.append((lnD, kc, snr_planck_TT, snr_planck_TTEE,
                          snr_litebird, snr_combined, snr_cmbs4))
    
    print(f"{lnD:12.1e} {kc:14.3e} {snr_planck_TT:12.2f}σ {snr_planck_TTEE:13.2f}σ "
          f"{snr_litebird:12.2f}σ {snr_combined:15.2f}σ {snr_cmbs4:10.2f}σ")


# ============================================================
# Cumulative SNR plot for a representative case
# ============================================================
outdir = '../../paper/figures'
os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14,
    'legend.fontsize': 10, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

# Use Contaldi-like lnD for illustration
lnD_demo = 2.2e13
cl_demo = compute_cl(lnD=lnD_demo, alpha=alpha_fid)
kc_demo = kc_from_lnD(lnD_demo)

ells_p, cum_p, _ = fisher_snr(cl_lcdm, cl_demo, planck_fsky,
                               planck_noise_TT_muK2, planck_noise_EE_muK2,
                               use_TT=True, use_EE=False, ell_max=50)
ells_pe, cum_pe, _ = fisher_snr(cl_lcdm, cl_demo, planck_fsky,
                                 planck_noise_TT_muK2, planck_noise_EE_muK2,
                                 use_TT=True, use_EE=True, ell_max=50)
ells_lb, cum_lb, _ = fisher_snr(cl_lcdm, cl_demo, litebird_fsky,
                                 0, litebird_noise_EE_muK2,
                                 use_TT=False, use_EE=True, ell_max=50)
# Combined cumulative
cum_combined = np.sqrt(cum_p**2 + cum_lb**2)

ells_s4, cum_s4, _ = fisher_snr(cl_lcdm, cl_demo, cmbs4_fsky,
                                 0, cmbs4_noise_EE_muK2,
                                 use_TT=False, use_EE=True, ell_max=50)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ells_p, cum_p, 'b-', lw=2, label='Planck TT only')
ax.plot(ells_pe, cum_pe, 'b--', lw=2, label='Planck TT+EE')
ax.plot(ells_lb, cum_lb, 'r-', lw=2, label='LiteBIRD EE only')
ax.plot(ells_p, cum_combined, 'k-', lw=2.5, label='Planck TT + LiteBIRD EE')
ax.plot(ells_s4, cum_s4, 'g-', lw=2, label='CMB-S4 EE only')

ax.axhline(1, color='gray', ls=':', lw=1, alpha=0.7)
ax.axhline(2, color='gray', ls=':', lw=1, alpha=0.7)
ax.axhline(3, color='gray', ls=':', lw=1, alpha=0.7)
ax.text(51, 1.0, r'$1\sigma$', fontsize=10, color='gray', va='center')
ax.text(51, 2.0, r'$2\sigma$', fontsize=10, color='gray', va='center')
ax.text(51, 3.0, r'$3\sigma$', fontsize=10, color='gray', va='center')

ax.set_xlabel(r'Maximum multipole $\ell_{\rm max}$')
ax.set_ylabel(r'Cumulative detection significance [$\sigma$]')
ax.set_title(rf'Fisher Forecast: CCR Detection Significance ($\ln D = 2.2\times10^{{13}}$, '
             rf'$k_c = {kc_demo:.1e}$ Mpc$^{{-1}}$, $\alpha = 2$)')
ax.set_xlim(2, 50)
ax.set_ylim(0, max(cum_combined.max(), cum_s4.max()) * 1.2)
ax.legend(loc='lower right', fontsize=10)
ax.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig_fisher_forecast.png')
plt.savefig(f'{outdir}/fig_fisher_forecast.pdf')
plt.close()
print(f"\nSaved: fig_fisher_forecast")


# ============================================================
# SNR vs ln D plot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

lnD_arr = np.array([r[0] for r in results_table])
snr_pTT = np.array([r[2] for r in results_table])
snr_pTTEE = np.array([r[3] for r in results_table])
snr_lb = np.array([r[4] for r in results_table])
snr_comb = np.array([r[5] for r in results_table])
snr_s4 = np.array([r[6] for r in results_table])

ax.plot(lnD_arr, snr_pTT, 'bo-', lw=2, markersize=6, label='Planck TT only')
ax.plot(lnD_arr, snr_pTTEE, 'b^--', lw=2, markersize=6, label='Planck TT+EE')
ax.plot(lnD_arr, snr_lb, 'rs-', lw=2, markersize=6, label='LiteBIRD EE only')
ax.plot(lnD_arr, snr_comb, 'kD-', lw=2.5, markersize=7, label='Planck TT + LiteBIRD EE')
ax.plot(lnD_arr, snr_s4, 'gv-', lw=2, markersize=6, label='CMB-S4 EE only')

ax.axhline(2, color='gray', ls=':', lw=1)
ax.axhline(3, color='gray', ls=':', lw=1)
ax.text(3e14, 2.1, r'$2\sigma$', fontsize=10, color='gray')
ax.text(3e14, 3.1, r'$3\sigma$', fontsize=10, color='gray')

ax.set_xscale('log')
ax.set_xlabel(r'$\ln D$')
ax.set_ylabel(r'Detection significance [$\sigma$]')
ax.set_title(r'CCR Detection Reach vs Hilbert Space Dimension ($\alpha = 2$, $\ell_{\rm max} = 30$)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(3e12, 3e14)

plt.savefig(f'{outdir}/fig_fisher_vs_lnD.png')
plt.savefig(f'{outdir}/fig_fisher_vs_lnD.pdf')
plt.close()
print(f"Saved: fig_fisher_vs_lnD")


# ============================================================
# Improvement factor summary
# ============================================================
print("\n" + "=" * 60)
print("IMPROVEMENT FACTORS (LiteBIRD over Planck)")
print("=" * 60)
for lnD, kc, sp, spe, sl, sc, ss4 in results_table:
    if sp > 0:
        improvement = sc / sp
    else:
        improvement = float('inf')
    print(f"  lnD = {lnD:.1e}: Planck TT = {sp:.2f}σ, "
          f"Planck+LiteBIRD = {sc:.2f}σ, improvement = {improvement:.1f}×")

print("\n[DONE]")