#!/usr/bin/env python3
"""
Prior Sensitivity Analysis via Importance Reweighting
=====================================================
Reweights existing MCMC chains with alternative priors on
log10(ln D) and alpha to demonstrate robustness of results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# Load chain
# ============================================================
data = np.loadtxt('chains/ccr_mcmc.1.txt')
weights = data[:, 0]
log_lnD = data[:, 2]   # log10(ln D)
alpha = data[:, 3]      # alpha_ccr
ombh2 = data[:, 4]
omch2 = data[:, 5]
H0 = data[:, 6]
tau = data[:, 7]
As = data[:, 8]
ns = data[:, 9]
kc = data[:, 11]

n_samples = len(weights)
total_weight = np.sum(weights)
print(f"Chain: {n_samples} rows, total weight = {total_weight:.0f}")
print(f"log10(lnD) range: [{log_lnD.min():.2f}, {log_lnD.max():.2f}]")
print(f"alpha range: [{alpha.min():.2f}, {alpha.max():.2f}]")

# ============================================================
# Define prior configurations
# ============================================================
# Baseline: log10(lnD) ~ U[12, 16], alpha ~ U[1, 6]
# Alternative priors to test:

prior_configs = {
    'Baseline [12,16]\u00d7[1,6]': {
        'lnD_range': (12, 16), 'alpha_range': (1, 6),
        'label': 'Baseline'
    },
    'Narrow lnD [13,15]': {
        'lnD_range': (13, 15), 'alpha_range': (1, 6),
        'label': 'Narrow ln D: [13,15]'
    },
    'Wide lnD [11,17]': {
        'lnD_range': (11, 17), 'alpha_range': (1, 6),
        'label': r'Wide ln D: [11,17]'
    },
    'Narrow \u03b1 [1,4]': {
        'lnD_range': (12, 16), 'alpha_range': (1, 4),
        'label': r'Narrow $\alpha$: [1,4]'
    },
    'Wide \u03b1 [0.5,8]': {
        'lnD_range': (12, 16), 'alpha_range': (0.5, 8),
        'label': r'Wide $\alpha$: [0.5,8]'
    },
    'Both narrow [13,15]\u00d7[1,4]': {
        'lnD_range': (13, 15), 'alpha_range': (1, 4),
        'label': r'Both narrow [13,15]$\times$[1,4]'
    },
}


def importance_reweight(weights, log_lnD, alpha, lnD_range, alpha_range,
                        base_lnD=(12, 16), base_alpha=(1, 6)):
    """
    Importance reweighting: new_weight = old_weight * (new_prior / old_prior).
    For uniform priors, this is just truncation + rescaling.
    """
    new_weights = weights.copy()
    
    # Truncate samples outside new prior
    mask = ((log_lnD >= lnD_range[0]) & (log_lnD <= lnD_range[1]) &
            (alpha >= alpha_range[0]) & (alpha <= alpha_range[1]))
    new_weights[~mask] = 0
    
    # Reweight by prior ratio (uniform -> uniform is just volume ratio)
    # new_prior_density / old_prior_density
    old_volume = (base_lnD[1] - base_lnD[0]) * (base_alpha[1] - base_alpha[0])
    new_volume = (lnD_range[1] - lnD_range[0]) * (alpha_range[1] - alpha_range[0])
    new_weights[mask] *= old_volume / new_volume
    
    return new_weights


def weighted_stats(values, weights):
    """Compute weighted mean and std."""
    mask = weights > 0
    if mask.sum() == 0:
        return np.nan, np.nan
    w = weights[mask]
    v = values[mask]
    mean = np.average(v, weights=w)
    var = np.average((v - mean)**2, weights=w)
    return mean, np.sqrt(var)


# ============================================================
# Compute stats for each prior configuration
# ============================================================
print("\n" + "=" * 110)
print(f"{'Prior configuration':<45s} {'<log10(lnD)>':>14s} {'<alpha>':>10s} "
      f"{'<H0>':>10s} {'<ns>':>10s} {'<tau>':>10s} {'N_eff':>8s}")
print("-" * 110)

results = []
for name, cfg in prior_configs.items():
    new_w = importance_reweight(weights, log_lnD, alpha,
                                 cfg['lnD_range'], cfg['alpha_range'])
    
    m_lnD, s_lnD = weighted_stats(log_lnD, new_w)
    m_alpha, s_alpha = weighted_stats(alpha, new_w)
    m_H0, s_H0 = weighted_stats(H0, new_w)
    m_ns, s_ns = weighted_stats(ns, new_w)
    m_tau, s_tau = weighted_stats(tau, new_w)
    
    # Effective sample size
    w_nonzero = new_w[new_w > 0]
    n_eff = np.sum(w_nonzero)**2 / np.sum(w_nonzero**2)
    
    results.append({
        'name': name, 'cfg': cfg,
        'label': cfg['label'],
        'lnD': (m_lnD, s_lnD), 'alpha': (m_alpha, s_alpha),
        'H0': (m_H0, s_H0), 'ns': (m_ns, s_ns), 'tau': (m_tau, s_tau),
        'n_eff': n_eff, 'weights': new_w
    })
    
    # Clean name for printing
    pname = name.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
    print(f"{pname:<45s} {m_lnD:>7.2f}±{s_lnD:<5.2f} {m_alpha:>5.2f}±{s_alpha:<4.2f} "
          f"{m_H0:>7.2f}±{s_H0:<4.2f} {m_ns:>7.4f}±{s_ns:<6.4f} {m_tau:>6.4f}±{s_tau:<6.4f} {n_eff:>7.0f}")

# ============================================================
# Key test: do ΛCDM parameters shift?
# ============================================================
print("\n" + "=" * 80)
print("STABILITY OF ΛCDM PARAMETERS ACROSS PRIOR CHOICES")
print("=" * 80)

baseline = results[0]
print(f"\n{'Prior':<35s} {'ΔH0':>8s} {'Δns':>10s} {'Δtau':>10s}")
print("-" * 65)
for r in results:
    dH0 = abs(r['H0'][0] - baseline['H0'][0])
    dns = abs(r['ns'][0] - baseline['ns'][0])
    dtau = abs(r['tau'][0] - baseline['tau'][0])
    pname = r['name'][:35].replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
    print(f"{pname:<35s} {dH0:>8.3f} {dns:>10.5f} {dtau:>10.5f}")


# ============================================================
# Plot: posterior comparison
# ============================================================
outdir = './figures'
os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14,
    'legend.fontsize': 9, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors for prior configs
colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']

# Panel 1: log10(lnD) posterior
ax = axes[0]
bins_lnD = np.linspace(12, 16, 40)
for i, r in enumerate(results):
    w = r['weights']
    mask = w > 0
    ax.hist(log_lnD[mask], bins=bins_lnD, weights=w[mask], density=True,
            histtype='step', lw=2, color=colors[i], label=r['label'])
ax.set_xlabel(r'$\log_{10}(\ln D)$')
ax.set_ylabel('Posterior density')
ax.set_title(r'$\log_{10}(\ln D)$ posterior')
ax.legend(fontsize=7, loc='upper left')

# Panel 2: alpha posterior
ax = axes[1]
bins_alpha = np.linspace(1, 6, 30)
for i, r in enumerate(results):
    w = r['weights']
    mask = w > 0
    valid = mask & (alpha >= 1) & (alpha <= 6)
    if valid.sum() > 0:
        ax.hist(alpha[valid], bins=bins_alpha, weights=w[valid], density=True,
                histtype='step', lw=2, color=colors[i])
ax.set_xlabel(r'$\alpha$')
ax.set_title(r'$\alpha$ posterior')

# Panel 3: H0 posterior (should be identical)
ax = axes[2]
bins_H0 = np.linspace(65.5, 69, 35)
for i, r in enumerate(results):
    w = r['weights']
    mask = w > 0
    ax.hist(H0[mask], bins=bins_H0, weights=w[mask], density=True,
            histtype='step', lw=2, color=colors[i])
ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
ax.set_title(r'$H_0$ posterior (stability test)')

plt.suptitle('Prior Sensitivity Analysis: Posterior Stability Under Alternative Priors',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig_prior_sensitivity.png')
plt.savefig(f'{outdir}/fig_prior_sensitivity.pdf')
plt.close()
print(f"\nSaved: fig_prior_sensitivity")

print("\n[DONE]")