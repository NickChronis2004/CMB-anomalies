#!/usr/bin/env python3
"""
analyse_ccr_chains.py — Analyse CCR MCMC chains and produce plots.

Usage:
    python analyse_ccr_chains.py --chains-dir ./chains --prefix ccr_mcmc

Reads Cobaya MCMC output and produces:
  - Parameter constraints table
  - Corner plots (CCR params, all params)
  - 1D posteriors for log_lnD, alpha_ccr
  - Best-fit C_ell comparison plot
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from getdist import MCSamples, loadMCSamples
from getdist import plots as gdplots


# ============================================================
# Physical constants (for k_c computation)
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

def kc_from_lnD(lnD, N=60, Treh_GeV=1e15):
    Treh_K = Treh_GeV * GEV_TO_K
    kc_phys = (1.0 / LP) * np.sqrt(np.pi / lnD)
    ai = np.exp(-N) * (GS_0 / GS_REH)**(1./3.) * (T0_CMB / Treh_K)
    return ai * kc_phys * MPC_IN_M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chains-dir', default='./chains')
    parser.add_argument('--prefix', default='ccr_mcmc')
    parser.add_argument('--output-dir', default='./figures')
    args = parser.parse_args()
    
    root = os.path.join(args.chains_dir, args.prefix)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CCR MCMC Chain Analysis")
    print("=" * 60)
    print(f"\nLoading chains from: {root}")
    
    # ============================================================
    # Load chains
    # ============================================================
    samples = loadMCSamples(root)
    
    print(f"Loaded {samples.numrows} samples")
    print(f"Parameters: {samples.getParamNames().list()}")
    
    # ============================================================
    # Parameter constraints
    # ============================================================
    print("\n" + "=" * 60)
    print("PARAMETER CONSTRAINTS (68% CL)")
    print("=" * 60)
    
    key_params = ['log_lnD', 'alpha_ccr', 'kc', 'H0', 'ombh2', 'omch2', 'tau', 'As', 'ns']
    
    for par in key_params:
        try:
            stats = samples.getInlineLatex(par, limit=1)
            mean = samples.mean(par)
            std = samples.std(par)
            print(f"  {par:12s}: {mean:.6g} +/- {std:.4g}  [{stats}]")
        except Exception as e:
            print(f"  {par:12s}: not found ({e})")
    
    # Best-fit
    print("\n--- Best-fit point ---")
    bf_idx = samples.getLikeStats().logLike_sample
    print(f"  Best-fit -logL = {bf_idx:.2f}")
    
    # Get best-fit parameter values
    bf_params = {}
    for par in key_params:
        try:
            bf_params[par] = samples.getBestFit().getParamDict().get(par)
        except:
            pass
    
    # Manual best-fit from chains
    loglikes = samples.loglikes
    if loglikes is not None:
        best_idx = np.argmin(loglikes)
        print(f"  Min -logL = {loglikes[best_idx]:.2f} at sample {best_idx}")
        for par in ['log_lnD', 'alpha_ccr', 'H0', 'ombh2', 'omch2', 'tau', 'As', 'ns']:
            try:
                val = samples.samples[best_idx, samples.index[par]]
                print(f"    {par} = {val:.6g}")
                bf_params[par] = val
            except:
                pass
    
    # ============================================================
    # Key derived quantities
    # ============================================================
    print("\n--- Key CCR Results ---")
    
    log_lnD_mean = samples.mean('log_lnD')
    log_lnD_std = samples.std('log_lnD')
    lnD_mean = 10**log_lnD_mean
    kc_mean = kc_from_lnD(lnD_mean)
    
    print(f"  log10(ln D) = {log_lnD_mean:.3f} +/- {log_lnD_std:.3f}")
    print(f"  ln D = {lnD_mean:.3e}")
    print(f"  k_c = {kc_mean:.3e} Mpc^-1")
    
    try:
        alpha_mean = samples.mean('alpha_ccr')
        alpha_std = samples.std('alpha_ccr')
        print(f"  alpha = {alpha_mean:.2f} +/- {alpha_std:.2f}")
    except:
        pass
    
    # Check if log_lnD hits prior boundary
    log_lnD_samples = samples.getParams().log_lnD
    pct_at_upper = np.sum(log_lnD_samples > 15.5) / len(log_lnD_samples) * 100
    pct_at_lower = np.sum(log_lnD_samples < 12.5) / len(log_lnD_samples) * 100
    print(f"\n  Prior boundary check:")
    print(f"    % near upper (>15.5): {pct_at_upper:.1f}%")
    print(f"    % near lower (<12.5): {pct_at_lower:.1f}%")
    if pct_at_upper > 10:
        print(f"    WARNING: Posterior piles up at upper boundary → data prefer NO cutoff (LCDM)")
    
    # ============================================================
    # Plots
    # ============================================================
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })
    
    # ---- 1. Corner plot: CCR params ----
    print("\n--- Making plots ---")
    try:
        g = gdplots.get_subplot_plotter(width_inch=8)
        g.triangle_plot(
            samples,
            ['log_lnD', 'alpha_ccr'],
            filled=True,
            title_limit=1,
            colors=['#2196F3'],
        )
        outpath = os.path.join(args.output_dir, 'corner_ccr_params.png')
        plt.savefig(outpath)
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  Corner CCR failed: {e}")
    
    # ---- 2. Corner plot: CCR + cosmological params ----
    try:
        g2 = gdplots.get_subplot_plotter(width_inch=12)
        g2.triangle_plot(
            samples,
            ['log_lnD', 'alpha_ccr', 'H0', 'ombh2', 'omch2', 'tau', 'ns'],
            filled=True,
            title_limit=1,
            colors=['#2196F3'],
        )
        outpath = os.path.join(args.output_dir, 'corner_all_params.png')
        plt.savefig(outpath)
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  Corner all failed: {e}")
    
    # ---- 3. 1D posteriors ----
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # log_lnD
        ax = axes[0]
        vals = samples.getParams().log_lnD
        ax.hist(vals, bins=80, density=True, alpha=0.7, color='#2196F3', edgecolor='white')
        ax.axvline(samples.mean('log_lnD'), color='red', ls='--', lw=2, label='Mean')
        ax.set_xlabel(r'$\log_{10}(\ln D)$')
        ax.set_ylabel('Posterior density')
        ax.set_title(r'$\log_{10}(\ln D)$')
        ax.legend()
        
        # alpha
        ax = axes[1]
        vals = samples.getParams().alpha_ccr
        ax.hist(vals, bins=80, density=True, alpha=0.7, color='#FF9800', edgecolor='white')
        ax.axvline(samples.mean('alpha_ccr'), color='red', ls='--', lw=2, label='Mean')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Posterior density')
        ax.set_title(r'$\alpha$')
        ax.legend()
        
        # kc (derived)
        ax = axes[2]
        try:
            vals = samples.getParams().kc
            ax.hist(vals, bins=80, density=True, alpha=0.7, color='#4CAF50', edgecolor='white')
            ax.axvline(np.median(vals), color='red', ls='--', lw=2, label='Median')
            ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]')
            ax.set_ylabel('Posterior density')
            ax.set_title(r'$k_c$')
            ax.legend()
        except:
            # Compute kc from log_lnD
            log_lnD_vals = samples.getParams().log_lnD
            kc_vals = np.array([kc_from_lnD(10**x) for x in log_lnD_vals])
            ax.hist(kc_vals, bins=80, density=True, alpha=0.7, color='#4CAF50', edgecolor='white')
            ax.axvline(np.median(kc_vals), color='red', ls='--', lw=2, label='Median')
            ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]')
            ax.set_ylabel('Posterior density')
            ax.set_title(r'$k_c$ (computed)')
            ax.legend()
        
        plt.tight_layout()
        outpath = os.path.join(args.output_dir, 'posteriors_1d.png')
        plt.savefig(outpath)
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  1D posteriors failed: {e}")
    
    # ---- 4. log_lnD posterior with interpretation ----
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = samples.getParams().log_lnD
        ax.hist(vals, bins=100, density=True, alpha=0.7, color='#2196F3', 
                edgecolor='white', label='Posterior')
        
        # Mark key values
        ax.axvline(np.log10(2e14), color='green', ls='--', lw=2, 
                   label=r'Fiducial ($\ln D = 2\times10^{14}$)')
        ax.axvline(np.log10(2.2e13), color='orange', ls='--', lw=2,
                   label=r'Contaldi ($k_c = 4.9\times10^{-4}$)')
        
        # Prior boundaries
        ax.axvline(12, color='gray', ls=':', lw=1)
        ax.axvline(16, color='gray', ls=':', lw=1)
        
        ax.set_xlabel(r'$\log_{10}(\ln D)$', fontsize=14)
        ax.set_ylabel('Posterior density', fontsize=14)
        ax.set_title('Posterior on Hilbert Space Dimension', fontsize=14)
        ax.legend(fontsize=10)
        
        outpath = os.path.join(args.output_dir, 'posterior_lnD_annotated.png')
        plt.savefig(outpath)
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  Annotated posterior failed: {e}")
    
    # ---- 5. Correlation: log_lnD vs LCDM params ----
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        log_lnD = samples.getParams().log_lnD
        
        for i, (par, label) in enumerate([
            ('H0', r'$H_0$'), ('ns', r'$n_s$'), ('tau', r'$\tau$')
        ]):
            ax = axes[i]
            par_vals = getattr(samples.getParams(), par)
            ax.scatter(log_lnD[::5], par_vals[::5], s=1, alpha=0.3, c='#2196F3')
            ax.set_xlabel(r'$\log_{10}(\ln D)$')
            ax.set_ylabel(label)
        
        plt.tight_layout()
        outpath = os.path.join(args.output_dir, 'correlations_lnD.png')
        plt.savefig(outpath)
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  Correlations failed: {e}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    log_lnD_mean = samples.mean('log_lnD')
    log_lnD_std = samples.std('log_lnD')
    
    print(f"\n  Chains: {samples.numrows} samples")
    print(f"  log10(ln D) = {log_lnD_mean:.2f} +/- {log_lnD_std:.2f}")
    print(f"  ln D = {10**log_lnD_mean:.2e}")
    print(f"  k_c = {kc_from_lnD(10**log_lnD_mean):.3e} Mpc^-1")
    
    try:
        print(f"  alpha = {samples.mean('alpha_ccr'):.2f} +/- {samples.std('alpha_ccr'):.2f}")
    except:
        pass
    
    # Interpretation
    log_lnD_vals = samples.getParams().log_lnD
    pct_above_15 = np.sum(log_lnD_vals > 15) / len(log_lnD_vals) * 100
    
    print(f"\n  % of posterior with log_lnD > 15 (LCDM-like): {pct_above_15:.1f}%")
    
    if pct_above_15 > 50:
        print("  → Data are CONSISTENT with LCDM (no cutoff needed)")
        print("  → We can place an UPPER BOUND on the cutoff scale")
    elif pct_above_15 < 10:
        print("  → Data PREFER a finite cutoff!")
        print("  → Evidence for CCR modification")
    else:
        print("  → MILD preference for cutoff, but LCDM not excluded")
    
    print(f"\n  Figures saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
