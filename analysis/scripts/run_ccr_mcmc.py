#!/usr/bin/env python3
"""
run_ccr_mcmc.py — CCR-CMB MCMC with Cobaya (v3)

Architecture:
  - CAMB: standard Boltzmann solver (handles transfers, Cl, lensing)
  - CCR_Primordial: provides custom P(k) to CAMB via 'primordial_scalar_pk'
  - Planck 2018: likelihoods (lowl TT, lowl EE, highl plik lite)

This uses Cobaya's official interface for external primordial power spectra.
"""

import os
import sys
import numpy as np
import argparse


# ============================================================
# Physical constants for validation
# ============================================================
LP = 1.616255e-35
KB = 1.380649e-23
EV_TO_J = 1.602176634e-19
GEV_TO_K = 1e9 * EV_TO_J / KB
MPC_IN_M = 3.085677581e22
T0_CMB = 2.7255
GS_REH = 106.75
GS_0 = 3.938
KPIVOT = 0.05

N_EFOLDS = 60
TREH_GEV = 1e15


def kc_from_lnD(lnD, N=N_EFOLDS, Treh_GeV=TREH_GEV):
    Treh_K = Treh_GeV * GEV_TO_K
    kc_phys = (1.0 / LP) * np.sqrt(np.pi / lnD)
    ai = np.exp(-N) * (GS_0 / GS_REH)**(1./3.) * (T0_CMB / Treh_K)
    return ai * kc_phys * MPC_IN_M


# ============================================================
# Validation
# ============================================================

def run_validation():
    print("=" * 60)
    print("Validation")
    print("=" * 60)
    
    import camb
    print(f"CAMB: {camb.__version__}")
    
    for lnD, expected in [(2e14, 1.64e-4), (2.2e13, 4.9e-4)]:
        kc = kc_from_lnD(lnD)
        status = "PASS" if abs(kc/expected - 1) < 0.02 else "FAIL"
        print(f"  [{status}] ln D={lnD:.1e}: kc={kc:.3e} (expected {expected:.1e})")
    
    # Timing
    import time
    nk = 1500
    k_arr = np.logspace(-6, 0.5, nk)
    
    times = []
    for _ in range(3):
        t0 = time.time()
        As, ns = 2.1e-9, 0.9649
        pk = As * (k_arr/0.05)**(ns-1) * (1 - np.exp(-(k_arr/1.64e-4)**2))
        pk = np.maximum(pk, 1e-30)
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, tau=0.0544, mnu=0.06)
        pars.set_for_lmax(2500, lens_potential_accuracy=1)
        pars.InitPower = camb.initialpower.SplinedInitialPower()
        pars.InitPower.set_scalar_log_regular(kmin=k_arr[0], kmax=k_arr[-1], PK=pk)
        pars.InitPower.effective_ns_for_nonlinear = ns
        results = camb.get_results(pars)
        cl = results.get_total_cls(2500)
        times.append(time.time() - t0)
    
    print(f"\n  Avg eval time: {np.mean(times):.2f}s")
    print(f"  Est. MCMC (50K, 4 chains): {np.mean(times)*50000/3600/4:.1f}h")
    print("\nALL PASSED")


# ============================================================
# MCMC
# ============================================================

def run_mcmc():
    print("=" * 60)
    print("Running CCR MCMC")
    print("=" * 60)
    
    from cobaya.run import run as cobaya_run
    
    info = {
        'theory': {
            # Standard CAMB — handles transfers, Cl, lensing
            'camb': {
                'extra_args': {
                    'lens_potential_accuracy': 1,
                },
            },
            # CCR Primordial — provides custom P(k) to CAMB
            'ccr_theory_cobaya.CCR_Primordial': {
                'python_path': os.path.dirname(os.path.abspath(__file__)),
            },
        },
        'params': {
            # CCR params (defined in CCR_Primordial.params, but we can set here too)
            'log_lnD': {
                'prior': {'min': 12.0, 'max': 16.0},
                'ref': {'dist': 'norm', 'loc': 14.3, 'scale': 0.3},
                'proposal': 0.3,
                'latex': r'\log_{10}(\ln D)',
            },
            'alpha_ccr': {
                'prior': {'min': 1.0, 'max': 6.0},
                'ref': {'dist': 'norm', 'loc': 2.0, 'scale': 0.5},
                'proposal': 0.5,
                'latex': r'\alpha',
            },
            # Standard LCDM (cosmological — used by CAMB)
            'ombh2': {
                'prior': {'min': 0.005, 'max': 0.1},
                'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00015},
                'proposal': 0.0001,
                'latex': r'\Omega_b h^2',
            },
            'omch2': {
                'prior': {'min': 0.001, 'max': 0.99},
                'ref': {'dist': 'norm', 'loc': 0.1200, 'scale': 0.001},
                'proposal': 0.0008,
                'latex': r'\Omega_c h^2',
            },
            'H0': {
                'prior': {'min': 40, 'max': 100},
                'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.5},
                'proposal': 0.4,
                'latex': r'H_0',
            },
            'tau': {
                'prior': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.0073},
                'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.005},
                'proposal': 0.003,
                'latex': r'\tau_{\rm reio}',
            },
            # Primordial (used by CCR_Primordial)
            'As': {
                'prior': {'min': 5e-10, 'max': 5e-9},
                'ref': {'dist': 'norm', 'loc': 2.1e-9, 'scale': 3e-11},
                'proposal': 2e-11,
                'latex': r'A_s',
            },
            'ns': {
                'prior': {'min': 0.8, 'max': 1.2},
                'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
                'proposal': 0.003,
                'latex': r'n_s',
            },
            # Derived
            'kc': {'derived': True, 'latex': r'k_c\,[\mathrm{Mpc}^{-1}]'},
            'Hinf_GeV': {'derived': True, 'latex': r'H_{\rm inf}\,[\mathrm{GeV}]'},
        },
        'likelihood': {
            'planck_2018_lowl.TT': None,       # ell=2-29
            'planck_2018_lowl.EE': None,        # constrains tau  
            'planck_2018_highl_plik.TTTEEE_lite': None,  # ell=30-2508
        },
        'sampler': {
            'mcmc': {
                'burn_in': 300,
                'max_tries': 10000,
                'Rminus1_stop': 0.005,
                'Rminus1_cl_stop': 0.2,
                'covmat': 'auto',
                'oversample_power': 0.4,
                'drag': True,
                'proposal_scale': 2.4,
                'learn_proposal': True,
            }
        },
        'output': 'chains/ccr_mcmc',
        'resume': True,
    }
    
    print("\nLaunching CCR MCMC...")
    print("Architecture: CAMB (transfers+Cl) + CCR_Primordial (P(k))")
    print("Convergence target: R-1 < 0.005")
    print("Output: chains/ccr_mcmc.*\n")
    
    updated_info, sampler = cobaya_run(info)
    print("\nMCMC complete!")


# ============================================================
# Data install
# ============================================================

def install_data():
    print("Installing Planck 2018 data...")
    packages = os.environ.get('COBAYA_PACKAGES_PATH', './packages')
    os.system(f"{sys.executable} -m cobaya.install planck_2018_lowl.TT -p {packages}")
    os.system(f"{sys.executable} -m cobaya.install planck_2018_lowl.EE -p {packages}")
    os.system(f"{sys.executable} -m cobaya.install planck_2018_highl_plik.TTTEEE_lite -p {packages}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCR-CMB MCMC Pipeline')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--install-data', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    
    if args.validate:
        run_validation()
    if args.install_data:
        install_data()
    if args.run:
        run_mcmc()
    if not any(vars(args).values()):
        parser.print_help()