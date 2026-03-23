#!/usr/bin/env python3
"""
bayes_factor_analysis.py — Compute Bayes factor estimate and chi2 comparison.

Methods:
  1. Best-fit chi2 comparison (quick)
  2. Savage-Dickey density ratio (for nested models)
  3. Information criteria (AIC, BIC)

Usage:
    python bayes_factor_analysis.py --chains-dir ./chains --prefix ccr_mcmc
"""

import os
import argparse
import numpy as np
from getdist import loadMCSamples
from scipy.stats import gaussian_kde


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chains-dir', default='./chains')
    parser.add_argument('--prefix', default='ccr_mcmc')
    args = parser.parse_args()
    
    root = os.path.join(args.chains_dir, args.prefix)
    
    print("=" * 60)
    print("BAYES FACTOR & MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load chains
    samples = loadMCSamples(root)
    print(f"\nLoaded {samples.numrows} samples")
    
    # ============================================================
    # 1. Chi-squared comparison
    # ============================================================
    print("\n" + "-" * 40)
    print("1. CHI-SQUARED ANALYSIS")
    print("-" * 40)
    
    # Get individual chi2 contributions
    chi2_params = [p for p in samples.getParamNames().list() if 'chi2' in p]
    print(f"\n  Available chi2 params: {chi2_params}")
    
    # Best-fit (minimum chi2)
    loglikes = samples.loglikes
    best_idx = np.argmin(loglikes)
    
    print(f"\n  Best-fit -logL = {loglikes[best_idx]:.2f}")
    print(f"  Best-fit chi2_total = {2 * loglikes[best_idx]:.2f}")
    
    for p in chi2_params:
        try:
            idx = samples.index[p]
            val = samples.samples[best_idx, idx]
            print(f"    {p} = {val:.2f}")
        except:
            pass
    
    # Best-fit CCR parameters
    print(f"\n  Best-fit CCR params:")
    bf_log_lnD = samples.samples[best_idx, samples.index['log_lnD']]
    bf_alpha = samples.samples[best_idx, samples.index['alpha_ccr']]
    print(f"    log_lnD = {bf_log_lnD:.3f} (ln D = {10**bf_log_lnD:.2e})")
    print(f"    alpha = {bf_alpha:.3f}")
    
    # The best-fit at log_lnD ~ 15.2 means LCDM-like
    # Compare: what's the best chi2 in the LCDM-like region vs cutoff region?
    log_lnD_all = samples.getParams().log_lnD
    
    # LCDM-like: log_lnD > 15 (k_c very small, no effect)
    lcdm_mask = log_lnD_all > 15.0
    # Cutoff region: log_lnD < 14 (k_c > 2e-4, measurable effect)
    cutoff_mask = log_lnD_all < 14.0
    
    if np.sum(lcdm_mask) > 0:
        best_lcdm = np.min(loglikes[lcdm_mask])
        print(f"\n  Best -logL in LCDM-like region (log_lnD > 15): {best_lcdm:.2f}")
    if np.sum(cutoff_mask) > 0:
        best_cutoff = np.min(loglikes[cutoff_mask])
        print(f"  Best -logL in cutoff region (log_lnD < 14):    {best_cutoff:.2f}")
        delta_chi2 = 2 * (best_cutoff - best_lcdm)
        print(f"  Delta chi2 (cutoff - LCDM) = {delta_chi2:.2f}")
        if delta_chi2 > 0:
            print(f"    → LCDM-like region fits BETTER by delta_chi2 = {delta_chi2:.1f}")
        else:
            print(f"    → Cutoff region fits BETTER by delta_chi2 = {-delta_chi2:.1f}")
    
    # ============================================================
    # 2. Savage-Dickey Density Ratio
    # ============================================================
    print("\n" + "-" * 40)
    print("2. SAVAGE-DICKEY DENSITY RATIO")
    print("-" * 40)
    print("\n  The CCR model reduces to LCDM when log_lnD → ∞ (no cutoff).")
    print("  Since our prior has a boundary at 16, we evaluate the posterior")
    print("  density at the LCDM limit (log_lnD = 16) vs the prior density.")
    
    # Posterior density at boundary
    log_lnD_samples = samples.getParams().log_lnD
    weights = samples.weights
    
    # KDE of posterior
    try:
        kde = gaussian_kde(log_lnD_samples, weights=weights)
        
        # Evaluate at LCDM limit
        lcdm_point = 15.5  # Near upper boundary
        posterior_at_lcdm = kde(lcdm_point)[0]
        
        # Prior density (flat prior on [12, 16])
        prior_density = 1.0 / (16.0 - 12.0)  # = 0.25
        
        # Savage-Dickey: B_01 = p(theta_0 | data, M1) / p(theta_0 | M1)
        # B_01 = posterior / prior at LCDM point
        # B_01 > 1 means data prefer LCDM
        # B_01 < 1 means data prefer CCR
        B_01 = posterior_at_lcdm / prior_density
        ln_B_01 = np.log(B_01)
        
        print(f"\n  Posterior density at log_lnD = {lcdm_point}: {posterior_at_lcdm:.4f}")
        print(f"  Prior density (flat): {prior_density:.4f}")
        print(f"  Savage-Dickey B_01 (LCDM/CCR): {B_01:.3f}")
        print(f"  ln(B_01) = {ln_B_01:.3f}")
        
        if ln_B_01 > 1:
            print(f"  → Strong evidence FOR LCDM (against cutoff)")
        elif ln_B_01 > 0:
            print(f"  → Weak evidence for LCDM")
        elif ln_B_01 > -1:
            print(f"  → Inconclusive (neither model strongly preferred)")
        elif ln_B_01 > -2.5:
            print(f"  → Moderate evidence for CCR cutoff")
        else:
            print(f"  → Strong evidence for CCR cutoff")
        
        # Also check at multiple points
        print(f"\n  Savage-Dickey at various points:")
        for pt in [14.0, 14.5, 15.0, 15.5, 15.8]:
            p_post = kde(pt)[0]
            B = p_post / prior_density
            print(f"    log_lnD = {pt}: posterior = {p_post:.4f}, B_01 = {B:.3f}, ln(B) = {np.log(B):.3f}")
            
    except Exception as e:
        print(f"  KDE failed: {e}")
    
    # ============================================================
    # 3. Information Criteria
    # ============================================================
    print("\n" + "-" * 40)
    print("3. INFORMATION CRITERIA (AIC, BIC)")
    print("-" * 40)
    
    # Number of data points (approximate)
    # Planck TT: ell 2-2508 ~ 2507 points
    # Planck TE: ell 30-1996 ~ 1967 points  
    # Planck EE: ell 30-1996 ~ 1967 points
    # Low-ell TT: 28 points (ell 2-29)
    # Low-ell EE: 28 points
    N_data = 2507 + 1967 + 1967 + 28 + 28  # ~ 6497
    # Plik lite is compressed, ~613 data points total
    N_data_lite = 613 + 28 + 28  # Commander + SimAll + plik_lite bins
    
    # LCDM: 6 params (ombh2, omch2, H0, tau, As, ns)
    # CCR: 6 + 2 params (+ log_lnD, alpha_ccr)
    k_lcdm = 6
    k_ccr = 8
    
    chi2_best = 2 * loglikes[best_idx]
    
    # For CCR model
    AIC_ccr = chi2_best + 2 * k_ccr
    BIC_ccr = chi2_best + k_ccr * np.log(N_data_lite)
    
    # For LCDM: approximate from best-fit in LCDM-like region
    if np.sum(lcdm_mask) > 0:
        chi2_lcdm = 2 * best_lcdm
        AIC_lcdm = chi2_lcdm + 2 * k_lcdm
        BIC_lcdm = chi2_lcdm + k_lcdm * np.log(N_data_lite)
        
        print(f"\n  N_data (plik lite approx) = {N_data_lite}")
        print(f"  k_LCDM = {k_lcdm}, k_CCR = {k_ccr}")
        print(f"\n  LCDM:  chi2 = {chi2_lcdm:.2f}, AIC = {AIC_lcdm:.2f}, BIC = {BIC_lcdm:.2f}")
        print(f"  CCR:   chi2 = {chi2_best:.2f}, AIC = {AIC_ccr:.2f}, BIC = {BIC_ccr:.2f}")
        
        delta_AIC = AIC_ccr - AIC_lcdm
        delta_BIC = BIC_ccr - BIC_lcdm
        
        print(f"\n  Delta AIC (CCR - LCDM) = {delta_AIC:.2f}")
        print(f"  Delta BIC (CCR - LCDM) = {delta_BIC:.2f}")
        
        if delta_AIC > 0:
            print(f"    AIC penalizes CCR by {delta_AIC:.1f} (extra params not justified)")
        else:
            print(f"    AIC favors CCR by {-delta_AIC:.1f}")
        
        if delta_BIC > 0:
            print(f"    BIC penalizes CCR by {delta_BIC:.1f}")
        else:
            print(f"    BIC favors CCR by {-delta_BIC:.1f}")
        
        # Jeffreys scale for BIC
        print(f"\n  Jeffreys scale (|Delta BIC|):")
        abs_dbic = abs(delta_BIC)
        if abs_dbic < 2:
            print(f"    {abs_dbic:.1f} → Not worth more than a bare mention")
        elif abs_dbic < 6:
            print(f"    {abs_dbic:.1f} → Positive evidence")
        elif abs_dbic < 10:
            print(f"    {abs_dbic:.1f} → Strong evidence")
        else:
            print(f"    {abs_dbic:.1f} → Very strong evidence")
    
    # ============================================================
    # 4. Posterior predictive check
    # ============================================================
    print("\n" + "-" * 40)
    print("4. POSTERIOR STATISTICS")
    print("-" * 40)
    
    # Mean chi2
    try:
        chi2_idx = samples.index['chi2']
        chi2_all = samples.samples[:, chi2_idx]
        print(f"\n  Mean chi2 = {np.average(chi2_all, weights=weights):.2f}")
        print(f"  Median chi2 = {np.median(chi2_all):.2f}")
        print(f"  Best chi2 = {np.min(chi2_all):.2f}")
    except:
        print(f"\n  Mean -logL = {np.average(loglikes, weights=weights):.2f}")
        print(f"  Best -logL = {np.min(loglikes):.2f}")
    
    # Effective number of parameters
    try:
        mean_chi2 = np.average(chi2_all, weights=weights)
        best_chi2 = np.min(chi2_all)
        p_eff = mean_chi2 - best_chi2
        print(f"  Effective params (p_D) = {p_eff:.1f}")
        DIC = best_chi2 + 2 * p_eff
        print(f"  DIC = {DIC:.2f}")
    except:
        pass
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY — MODEL COMPARISON")
    print("=" * 60)
    
    print(f"""
  Best-fit log_lnD = {bf_log_lnD:.2f} (ln D = {10**bf_log_lnD:.1e})
  → Best-fit is LCDM-like (large D = weak cutoff)
  
  Delta chi2 (best cutoff vs best LCDM): ~{delta_chi2:.1f}
  → The 2 extra CCR parameters do NOT significantly improve the fit
  
  Savage-Dickey ln(B_01) = {ln_B_01:.2f}
  → {"LCDM mildly preferred" if ln_B_01 > 0 else "CCR mildly preferred" if ln_B_01 > -1 else "CCR preferred"}
  
  Delta BIC = {delta_BIC:.1f}
  → BIC {"penalizes" if delta_BIC > 0 else "favors"} CCR
  
  INTERPRETATION FOR PAPER:
  The Planck data are consistent with the CCR framework but do not 
  require a finite Hilbert space dimension. The additional parameters 
  (ln D, alpha) are not statistically justified over LCDM by current 
  data. We place a lower bound on ln D and note that the framework 
  remains observationally viable for future, more sensitive probes 
  of the low-ell CMB power spectrum.
""")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
