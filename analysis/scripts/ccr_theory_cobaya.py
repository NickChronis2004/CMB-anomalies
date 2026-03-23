"""
ccr_theory_cobaya.py — CCR Primordial Power Spectrum for Cobaya (v3)

This provides a custom primordial P(k) to Cobaya's built-in CAMB theory
via the 'primordial' interface. CAMB handles everything else (transfers, 
Cl, lensing) and the Planck likelihood interface is automatically correct.

This is the officially supported way to inject custom P(k) in Cobaya.
"""

import numpy as np
from cobaya.theory import Theory

# Physical constants
LP = 1.616255e-35
KB = 1.380649e-23
EV_TO_J = 1.602176634e-19
GEV_TO_K = 1e9 * EV_TO_J / KB
MPC_IN_M = 3.085677581e22
T0_CMB = 2.7255
GS_REH = 106.75
GS_0 = 3.938
KPIVOT = 0.05
HBAR_C = 1.9733e-16


def kc_from_lnD(lnD, N=60, Treh_GeV=1e15):
    """Comoving IR cutoff k_c [Mpc^-1] from ln D."""
    Treh_K = Treh_GeV * GEV_TO_K
    kc_phys = (1.0 / LP) * np.sqrt(np.pi / lnD)
    ai = np.exp(-N) * (GS_0 / GS_REH) ** (1. / 3.) * (T0_CMB / Treh_K)
    return ai * kc_phys * MPC_IN_M


class CCR_Primordial(Theory):
    """
    Provides CCR-modified primordial scalar P(k) to CAMB via Cobaya.
    
    P_CCR(k) = A_s (k/k*)^(n_s-1) * [1 - exp(-(k/k_c)^alpha)]
    
    Sampled: log_lnD, alpha_ccr, As, ns
    Derived: kc, Hinf_GeV
    """
    
    params = {
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
        'kc': {
            'derived': True,
            'latex': r'k_c\,[\mathrm{Mpc}^{-1}]',
        },
        'Hinf_GeV': {
            'derived': True,
            'latex': r'H_{\rm inf}\,[\mathrm{GeV}]',
        },
    }
    
    nk = 1500
    kmin = 1e-6
    kmax = 3.2  # Mpc^-1
    
    def initialize(self):
        self.k_arr = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)
        self.log.info(
            f"CCR_Primordial initialized: k=[{self.kmin:.0e},{self.kmax:.1f}] "
            f"Mpc^-1 ({self.nk} pts)"
        )
    
    def get_can_provide(self):
        return ['primordial_scalar_pk']
    
    def get_can_provide_params(self):
        return ['kc', 'Hinf_GeV']
    
    def calculate(self, state, want_derived=True, **params_values):
        log_lnD = params_values['log_lnD']
        alpha = params_values['alpha_ccr']
        As = params_values['As']
        ns = params_values['ns']
        
        lnD = 10.0 ** log_lnD
        kc = kc_from_lnD(lnD)
        
        # Build CCR P(k)
        pk = As * (self.k_arr / KPIVOT) ** (ns - 1)
        pk *= (1.0 - np.exp(-(self.k_arr / kc) ** alpha))
        pk = np.maximum(pk, 1e-30)
        
        state['primordial_scalar_pk'] = {
            'log_regular': True,
            'kmin': self.k_arr[0],
            'kmax': self.k_arr[-1],
            'Pk': pk,
            'effective_ns_for_nonlinear': ns,
        }
        
        state['derived'] = {
            'kc': kc,
            'Hinf_GeV': (1.0 / LP) * np.sqrt(np.pi / lnD) * HBAR_C,
        }
    
    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']


if __name__ == '__main__':
    print("CCR_Primordial v3 — Primordial P(k) provider for Cobaya CAMB")
    print(f"kc(2e14) = {kc_from_lnD(2e14):.4e} Mpc^-1 (expected ~1.64e-4)")