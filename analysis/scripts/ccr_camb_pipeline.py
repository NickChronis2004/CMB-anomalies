#!/usr/bin/env python3
"""
CCR-CAMB Pipeline: Compute C_ell from CCR-modified primordial power spectrum.
Paper A — Phase 3: Computational Simulations
"""

import camb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

# ============================================================
# Physical constants and fiducial parameters
# ============================================================
lP = 1.616255e-35      # Planck length [m]
kB = 1.380649e-23      # Boltzmann constant [J/K]
eV_to_J = 1.602176634e-19
GeV_to_K = 1e9 * eV_to_J / kB  # 1 GeV in Kelvin
Mpc_in_m = 3.085677581e22       # 1 Mpc in meters

# Planck 2018 best-fit LCDM (Table 2 of Planck 2018 VI)
H0 = 67.36
ombh2 = 0.02237
omch2 = 0.1200
tau = 0.0544
As = 2.1e-9
ns = 0.9649
kpivot = 0.05  # Mpc^-1

# CCR thermal history
N_fid = 60
Treh_GeV = 1e15
Treh_K = Treh_GeV * GeV_to_K
T0 = 2.7255  # K
gSreh = 106.75
gS0 = 3.938

# k grid for primordial spectrum
nk = 1500
k_arr = np.logspace(-6, 0.5, nk)  # 10^-6 to ~3 Mpc^-1
pk_lcdm = As * (k_arr / kpivot)**(ns - 1)

lmax = 2500

# ============================================================
# Core functions
# ============================================================

def kc_from_lnD(lnD, N=N_fid, Treh_GeV_val=Treh_GeV):
    """Compute comoving IR cutoff k_c [Mpc^-1] from ln D via eq (11)."""
    Treh_K_val = Treh_GeV_val * GeV_to_K
    kc_phys = (1.0 / lP) * np.sqrt(np.pi / lnD)  # m^-1
    ai = np.exp(-N) * (gS0/gSreh)**(1./3.) * (T0 / Treh_K_val)
    kc = ai * kc_phys * Mpc_in_m  # Mpc^-1
    return kc


def ccr_pk(k, lnD, alpha, N=N_fid):
    """CCR-modified primordial power spectrum P_CCR(k)."""
    kc = kc_from_lnD(lnD, N)
    pk_std = As * (k / kpivot)**(ns - 1)
    suppression = 1.0 - np.exp(-(k / kc)**alpha)
    return pk_std * suppression, kc


def compute_cl(lnD=None, alpha=None, custom_pk=None):
    """
    Compute C_ell for CCR model (if lnD given) or LCDM (if lnD=None).
    Returns D_ell in muK^2, shape (lmax+1, 4) for TT, EE, BB, TE.
    """
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
# Compute everything
# ============================================================

print("=" * 60)
print("CCR-CAMB Pipeline — Phase 3")
print("=" * 60)

# 1. LCDM baseline
print("\n[1] Computing LCDM baseline...")
cl_lcdm, _ = compute_cl()
ells = np.arange(cl_lcdm.shape[0])
print(f"    Done. lmax = {lmax}")

# 2. CCR for grid of (lnD, alpha) values
lnD_values = [1e13, 2.2e13, 5e13, 1e14, 2e14, 5e14, 1e15]
alpha_values = [1, 2, 3, 4, 6]

# Fiducial scan over lnD at alpha=2
print("\n[2] Computing CCR C_ell for various ln D (alpha=2)...")
cl_lnD_scan = {}
for lnD in lnD_values:
    cl, kc = compute_cl(lnD=lnD, alpha=2)
    cl_lnD_scan[lnD] = (cl, kc)
    print(f"    ln D = {lnD:.1e}: k_c = {kc:.3e} Mpc^-1, C_2/C_2^LCDM = {cl[2,0]/cl_lcdm[2,0]:.4f}")

# Fiducial scan over alpha at lnD = 2e14
print("\n[3] Computing CCR C_ell for various alpha (ln D = 2e14)...")
cl_alpha_scan = {}
for alpha in alpha_values:
    cl, kc = compute_cl(lnD=2e14, alpha=alpha)
    cl_alpha_scan[alpha] = (cl, kc)
    print(f"    alpha = {alpha}: C_2/C_2^LCDM = {cl[2,0]/cl_lcdm[2,0]:.4f}, C_5 ratio = {cl[5,0]/cl_lcdm[5,0]:.4f}")

# 3. Contaldi best-fit comparison: kc = 4.9e-4 Mpc^-1
# Find lnD that gives kc = 4.9e-4
# kc = (e^-N / lP) * (gS0/gSreh)^(1/3) * (T0/Treh) * sqrt(pi/lnD)
# => lnD = pi * [(e^-N / lP) * (gS0/gSreh)^(1/3) * (T0/Treh) / kc_target]^2 / Mpc_in_m^2
# Easier: just invert numerically
from scipy.optimize import brentq
def kc_minus_target(lnD, target):
    return kc_from_lnD(lnD) - target

lnD_contaldi = brentq(kc_minus_target, 1e12, 1e16, args=(4.9e-4,))
print(f"\n[4] Contaldi best-fit k_c = 4.9e-4 Mpc^-1 => ln D = {lnD_contaldi:.3e}")
cl_contaldi, _ = compute_cl(lnD=lnD_contaldi, alpha=2)

# Sanity check: D -> infinity
print("\n[5] Sanity check: D -> infinity...")
cl_inf, kc_inf = compute_cl(lnD=1e20, alpha=2)
max_diff = max(abs(cl_inf[ell, 0] / cl_lcdm[ell, 0] - 1) for ell in range(2, 2501))
print(f"    k_c = {kc_inf:.3e} Mpc^-1")
print(f"    Max |C_ell^CCR/C_ell^LCDM - 1| = {max_diff:.2e}")

# ============================================================
# Plotting
# ============================================================

outdir = './figures'
os.makedirs(outdir, exist_ok=True)

# Common style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ---- FIGURE 4: TT power spectrum — LCDM vs CCR for various ln D ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={'hspace': 0.05})

ax1.plot(ells[2:], cl_lcdm[2:, 0], 'k-', lw=2, label=r'$\Lambda$CDM', zorder=10)

colors_lnD = plt.cm.viridis(np.linspace(0.15, 0.85, len(lnD_values)))
for i, lnD in enumerate(lnD_values):
    cl, kc = cl_lnD_scan[lnD]
    label = rf'$\ln D = {lnD:.0e}$, $k_c = {kc:.1e}$'
    ax1.plot(ells[2:], cl[2:, 0], color=colors_lnD[i], lw=1.2, alpha=0.8, label=label)

ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
ax1.set_xlim(2, 50)
ax1.set_ylim(0, 1400)
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.set_title(r'CCR-modified TT power spectrum ($\alpha = 2$, fiducial parameters)')
ax1.xaxis.set_minor_locator(AutoMinorLocator())

# Residuals
for i, lnD in enumerate(lnD_values):
    cl, kc = cl_lnD_scan[lnD]
    ratio = cl[2:51, 0] / cl_lcdm[2:51, 0]
    ax2.plot(ells[2:51], ratio, color=colors_lnD[i], lw=1.2, alpha=0.8)

ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.set_xlabel(r'Multipole $\ell$')
ax2.set_ylabel(r'$\mathcal{D}_\ell^{\rm CCR} / \mathcal{D}_\ell^{\Lambda{\rm CDM}}$')
ax2.set_ylim(0.3, 1.05)
ax2.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig4_cl_vs_lnD.png')
plt.close()
print(f"\nSaved: fig4_cl_vs_lnD.png")

# ---- FIGURE 5: TT power spectrum — effect of alpha ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={'hspace': 0.05})

ax1.plot(ells[2:], cl_lcdm[2:, 0], 'k-', lw=2, label=r'$\Lambda$CDM', zorder=10)

colors_alpha = plt.cm.plasma(np.linspace(0.15, 0.85, len(alpha_values)))
for i, alpha in enumerate(alpha_values):
    cl, kc = cl_alpha_scan[alpha]
    label = rf'$\alpha = {alpha}$'
    ax1.plot(ells[2:], cl[2:, 0], color=colors_alpha[i], lw=1.5, alpha=0.8, label=label)

ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
ax1.set_xlim(2, 50)
ax1.set_ylim(0, 1400)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title(rf'Effect of shape parameter $\alpha$ ($\ln D = 2 \times 10^{{14}}$, $k_c = 1.64 \times 10^{{-4}}$ Mpc$^{{-1}}$)')
ax1.xaxis.set_minor_locator(AutoMinorLocator())

# Residuals
for i, alpha in enumerate(alpha_values):
    cl, kc = cl_alpha_scan[alpha]
    ratio = cl[2:51, 0] / cl_lcdm[2:51, 0]
    ax2.plot(ells[2:51], ratio, color=colors_alpha[i], lw=1.5, alpha=0.8)

ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.set_xlabel(r'Multipole $\ell$')
ax2.set_ylabel(r'$\mathcal{D}_\ell^{\rm CCR} / \mathcal{D}_\ell^{\Lambda{\rm CDM}}$')
ax2.set_ylim(0.3, 1.05)
ax2.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig5_cl_vs_alpha.png')
plt.close()
print(f"Saved: fig5_cl_vs_alpha.png")

# ---- FIGURE 6: Full range TT spectrum with Contaldi comparison ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={'hspace': 0.05})

ax1.plot(ells[2:], cl_lcdm[2:, 0], 'k-', lw=1.5, label=r'$\Lambda$CDM')
cl_fid, kc_fid = cl_lnD_scan[2e14]
ax1.plot(ells[2:], cl_fid[2:, 0], 'b-', lw=1.5, 
         label=rf'CCR fiducial ($\ln D = 2\times10^{{14}}$, $k_c = {kc_fid:.2e}$)')
ax1.plot(ells[2:], cl_contaldi[2:, 0], 'r--', lw=1.5,
         label=rf'CCR at Contaldi $k_c$ ($\ln D = {lnD_contaldi:.2e}$, $k_c = 4.9\times10^{{-4}}$)')

ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
ax1.set_xscale('log')
ax1.set_xlim(2, 2500)
ax1.set_ylim(0, 6500)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_title(r'TT power spectrum: $\Lambda$CDM vs CCR ($\alpha = 2$)')

# Residuals
ratio_fid = cl_fid[2:, 0] / cl_lcdm[2:, 0]
ratio_cont = cl_contaldi[2:, 0] / cl_lcdm[2:, 0]
ax2.plot(ells[2:], ratio_fid, 'b-', lw=1.2, label='CCR fiducial')
ax2.plot(ells[2:], ratio_cont, 'r--', lw=1.2, label='CCR (Contaldi $k_c$)')
ax2.axhline(1, color='k', ls='--', lw=0.8)
ax2.set_xlabel(r'Multipole $\ell$')
ax2.set_ylabel(r'Ratio to $\Lambda$CDM')
ax2.set_ylim(0.5, 1.05)
ax2.legend(fontsize=9)

plt.savefig(f'{outdir}/fig6_cl_full_range.png')
plt.close()
print(f"Saved: fig6_cl_full_range.png")

# ---- FIGURE 7: Delta C_ell (difference) ----
fig, ax = plt.subplots(figsize=(10, 5))

for i, lnD in enumerate([1e13, 2.2e13, 5e13, 2e14]):
    cl, kc = cl_lnD_scan.get(lnD, (None, None))
    if cl is None:
        if lnD == 2.2e13:
            cl, kc = cl_contaldi, 4.9e-4
        else:
            continue
    delta = cl[2:51, 0] - cl_lcdm[2:51, 0]
    label = rf'$\ln D = {lnD:.1e}$'
    ax.plot(ells[2:51], delta, lw=1.5, label=label)

ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$\Delta \mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
ax.set_title(r'CMB power deficit: $\mathcal{D}_\ell^{\rm CCR} - \mathcal{D}_\ell^{\Lambda{\rm CDM}}$ ($\alpha = 2$)')
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f'{outdir}/fig7_delta_cl.png')
plt.close()
print(f"Saved: fig7_delta_cl.png")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: CCR suppression at key multipoles (alpha = 2)")
print("=" * 60)
print(f"{'ln D':>12s} {'k_c [Mpc^-1]':>14s} {'C_2 ratio':>10s} {'C_3 ratio':>10s} {'C_5 ratio':>10s} {'C_10 ratio':>10s}")
print("-" * 68)
for lnD in lnD_values:
    cl, kc = cl_lnD_scan[lnD]
    r2 = cl[2, 0] / cl_lcdm[2, 0]
    r3 = cl[3, 0] / cl_lcdm[3, 0]
    r5 = cl[5, 0] / cl_lcdm[5, 0]
    r10 = cl[10, 0] / cl_lcdm[10, 0]
    print(f"{lnD:12.1e} {kc:14.3e} {r2:10.4f} {r3:10.4f} {r5:10.4f} {r10:10.4f}")

print(f"\nContaldi best-fit: ln D = {lnD_contaldi:.3e}, k_c = 4.9e-4 Mpc^-1")
print(f"  C_2 ratio = {cl_contaldi[2,0]/cl_lcdm[2,0]:.4f}")

print("\n[DONE] All figures saved to", outdir)
