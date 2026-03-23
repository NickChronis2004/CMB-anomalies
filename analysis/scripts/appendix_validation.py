#!/usr/bin/env python3
"""
Appendix C: Numerical Validation — ΛCDM Recovery Test
"""
import camb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

H0 = 67.36; ombh2 = 0.02237; omch2 = 0.1200; tau = 0.0544
As = 2.1e-9; ns = 0.9649; kpivot = 0.05
lmax = 2500; nk = 1500
k_arr = np.logspace(-6, 0.5, nk)

outdir = './figures'
os.makedirs(outdir, exist_ok=True)

# ΛCDM baseline
pars0 = camb.CAMBparams()
pars0.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
pars0.InitPower.set_params(As=As, ns=ns)
pars0.set_for_lmax(lmax, lens_potential_accuracy=0)
res0 = camb.get_results(pars0)
cl0 = res0.get_cmb_power_spectra(pars0, CMB_unit='muK')['total']

# CCR with very large lnD (should recover ΛCDM)
test_lnDs = [1e14, 1e15, 1e16, 1e17, 1e18]
alpha = 2

# Physical constants
lP = 1.616255e-35; eV_to_J = 1.602176634e-19; kB = 1.380649e-23
GeV_to_K = 1e9 * eV_to_J / kB; Mpc_in_m = 3.085677581e22
N_fid = 60; Treh_GeV = 1e15; T0 = 2.7255; gSreh = 106.75; gS0 = 3.938

def kc_from_lnD(lnD):
    Treh_K = Treh_GeV * GeV_to_K
    kc_phys = (1.0/lP) * np.sqrt(np.pi/lnD)
    ai = np.exp(-N_fid) * (gS0/gSreh)**(1./3.) * (T0/Treh_K)
    return ai * kc_phys * Mpc_in_m

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})

ells = np.arange(2, lmax+1)

print(f"{'lnD':>12s}  {'kc [Mpc^-1]':>14s}  {'max |ΔCℓ/Cℓ| TT':>18s}  {'max |ΔCℓ/Cℓ| EE':>18s}")
print("-" * 70)

colors = ['red', 'orange', 'green', 'blue', 'purple']

for i, lnD in enumerate(test_lnDs):
    kc = kc_from_lnD(lnD)
    pk = As * (k_arr/kpivot)**(ns-1) * (1.0 - np.exp(-(k_arr/kc)**alpha))
    pk = np.maximum(pk, 1e-30)
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.InitPower = camb.initialpower.SplinedInitialPower()
    pars.InitPower.set_scalar_log_regular(kmin=k_arr[0], kmax=k_arr[-1], PK=pk)
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    
    # Fractional difference
    frac_TT = np.abs(cl[2:lmax+1, 0] - cl0[2:lmax+1, 0]) / np.maximum(cl0[2:lmax+1, 0], 1e-30)
    frac_EE = np.abs(cl[2:lmax+1, 1] - cl0[2:lmax+1, 1]) / np.maximum(cl0[2:lmax+1, 1], 1e-30)
    
    max_frac_TT = np.max(frac_TT)
    max_frac_EE = np.max(frac_EE)
    
    print(f"{lnD:12.1e}  {kc:14.3e}  {max_frac_TT:18.2e}  {max_frac_EE:18.2e}")
    
    axes[0].semilogy(ells, frac_TT, color=colors[i], lw=1.2, alpha=0.8,
                     label=rf'$\ln D = {lnD:.0e}$, $k_c = {kc:.1e}$ Mpc$^{{-1}}$')

axes[0].set_ylabel(r'$|\Delta C_\ell^{TT} / C_\ell^{TT}|$')
axes[0].set_title(r'$\Lambda$CDM Recovery Test: Fractional Difference $|C_\ell^{\rm CCR} - C_\ell^{\Lambda\rm CDM}| / C_\ell^{\Lambda\rm CDM}$')
axes[0].axhline(5e-5, color='gray', ls='--', lw=1, label=r'$5\times10^{-5}$ threshold')
axes[0].legend(fontsize=8, ncol=2, loc='upper right')
axes[0].set_ylim(1e-16, 1e-1)

# Grid convergence test
for nk_test in [500, 1000, 1500, 2000]:
    k_test = np.logspace(-6, 0.5, nk_test)
    pk_test = As * (k_test/kpivot)**(ns-1) * (1.0 - np.exp(-(k_test/kc_from_lnD(2e14))**2))
    pk_test = np.maximum(pk_test, 1e-30)
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.InitPower = camb.initialpower.SplinedInitialPower()
    pars.InitPower.set_scalar_log_regular(kmin=k_test[0], kmax=k_test[-1], PK=pk_test)
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    
    # Compare with nk=2000 as reference
    if nk_test == 2000:
        cl_ref = cl.copy()
    else:
        if 'cl_ref' in dir():
            frac = np.abs(cl[2:lmax+1, 0] - cl_ref[2:lmax+1, 0]) / np.maximum(cl_ref[2:lmax+1, 0], 1e-30)
            axes[1].semilogy(ells, frac, lw=1.2, label=f'$n_k = {nk_test}$ vs 2000')

# Rerun nk=500,1000,1500 after cl_ref exists
k_ref = np.logspace(-6, 0.5, 2000)
pk_ref = As * (k_ref/kpivot)**(ns-1) * (1.0 - np.exp(-(k_ref/kc_from_lnD(2e14))**2))
pk_ref = np.maximum(pk_ref, 1e-30)
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
pars.set_for_lmax(lmax, lens_potential_accuracy=0)
pars.InitPower = camb.initialpower.SplinedInitialPower()
pars.InitPower.set_scalar_log_regular(kmin=k_ref[0], kmax=k_ref[-1], PK=pk_ref)
res_ref = camb.get_results(pars)
cl_ref = res_ref.get_cmb_power_spectra(pars, CMB_unit='muK')['total']

for nk_test in [500, 1000, 1500]:
    k_test = np.logspace(-6, 0.5, nk_test)
    pk_test = As * (k_test/kpivot)**(ns-1) * (1.0 - np.exp(-(k_test/kc_from_lnD(2e14))**2))
    pk_test = np.maximum(pk_test, 1e-30)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.InitPower = camb.initialpower.SplinedInitialPower()
    pars.InitPower.set_scalar_log_regular(kmin=k_test[0], kmax=k_test[-1], PK=pk_test)
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    frac = np.abs(cl[2:lmax+1, 0] - cl_ref[2:lmax+1, 0]) / np.maximum(cl_ref[2:lmax+1, 0], 1e-30)
    axes[1].semilogy(ells, frac, lw=1.2, label=f'$n_k = {nk_test}$ vs 2000')

axes[1].set_xlabel(r'Multipole $\ell$')
axes[1].set_ylabel(r'$|\Delta C_\ell^{TT}|/C_\ell^{TT}$')
axes[1].set_title(r'$k$-Grid Convergence Test ($\ln D = 2\times10^{14}$, $\alpha = 2$)')
axes[1].axhline(1e-5, color='gray', ls='--', lw=1)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].set_ylim(1e-12, 1e-2)
axes[1].set_xlim(2, 2500)

plt.tight_layout()
plt.savefig(f'{outdir}/fig_lcdm_recovery.png')
plt.savefig(f'{outdir}/fig_lcdm_recovery.pdf')
plt.close()
print("\nSaved: fig_lcdm_recovery")
print("[DONE]")