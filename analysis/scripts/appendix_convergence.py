#!/usr/bin/env python3
"""
Appendix B: MCMC Convergence Diagnostics
Trace plots, running mean, autocorrelation for key parameters.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

data = np.loadtxt('chains/ccr_mcmc.1.txt')
weights = data[:, 0]
log_lnD = data[:, 2]
alpha = data[:, 3]
H0 = data[:, 6]
tau = data[:, 7]
ns = data[:, 9]
chi2 = data[:, 16]

n = len(weights)
cumulative_sample = np.arange(1, n+1)

outdir = './figures'
os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

# ============================================================
# Figure B1: Trace plots
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

params = [
    (log_lnD, r'$\log_{10}(\ln D)$', 'tab:blue'),
    (alpha, r'$\alpha$', 'tab:red'),
    (H0, r'$H_0$', 'tab:green'),
    (tau, r'$\tau$', 'tab:orange'),
    (ns, r'$n_s$', 'tab:purple'),
]

for ax, (param, label, color) in zip(axes, params):
    ax.plot(cumulative_sample, param, color=color, alpha=0.4, lw=0.3, rasterized=True)
    # Running mean (weighted)
    cum_w = np.cumsum(weights)
    cum_wp = np.cumsum(weights * param)
    running_mean = cum_wp / cum_w
    ax.plot(cumulative_sample, running_mean, 'k-', lw=1.5, label='Running mean')
    ax.set_ylabel(label)
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Sample number')
axes[0].set_title('MCMC Trace Plots and Running Means')
plt.tight_layout()
plt.savefig(f'{outdir}/fig_trace_plots.png')
plt.savefig(f'{outdir}/fig_trace_plots.pdf')
plt.close()
print("Saved: fig_trace_plots")

# ============================================================
# Figure B2: Running Gelman-Rubin proxy (split-chain R-1)
# ============================================================
# Split single chain into 2 halves, compute running variance ratio
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

test_params = [
    (log_lnD, r'$\log_{10}(\ln D)$'),
    (alpha, r'$\alpha$'),
    (H0, r'$H_0$'),
]

for ax, (param, label) in zip(axes, test_params):
    R_minus_1 = []
    check_points = np.arange(200, n, 50)
    
    for cp in check_points:
        chain = param[:cp]
        w = weights[:cp]
        mid = cp // 2
        
        # Split into two halves
        c1, w1 = chain[:mid], w[:mid]
        c2, w2 = chain[mid:], w[:mid]  # equal length
        
        m1 = np.average(c1, weights=w1)
        m2 = np.average(c2, weights=w2)
        v1 = np.average((c1-m1)**2, weights=w1)
        v2 = np.average((c2-m2)**2, weights=w2)
        
        W = (v1 + v2) / 2
        m_total = (m1 + m2) / 2
        B = ((m1 - m_total)**2 + (m2 - m_total)**2) / 1
        
        if W > 0:
            R = (W + B) / W
            R_minus_1.append(R - 1)
        else:
            R_minus_1.append(np.nan)
    
    ax.semilogy(check_points, R_minus_1, lw=1.5)
    ax.axhline(0.01, color='red', ls='--', lw=1, label='$R-1 = 0.01$ target')
    ax.axhline(0.012, color='orange', ls=':', lw=1, label='$R-1 = 0.012$ achieved')
    ax.set_ylabel(f'$R-1$ ({label})')
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Sample number')
axes[0].set_title('Split-Chain Gelman--Rubin Diagnostic ($R-1$)')
plt.tight_layout()
plt.savefig(f'{outdir}/fig_gelman_rubin.png')
plt.savefig(f'{outdir}/fig_gelman_rubin.pdf')
plt.close()
print("Saved: fig_gelman_rubin")

# ============================================================
# Figure B3: chi^2 trace
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(cumulative_sample, chi2, color='gray', alpha=0.4, lw=0.3, rasterized=True)
cum_w = np.cumsum(weights)
cum_wc = np.cumsum(weights * chi2)
running_chi2 = cum_wc / cum_w
ax.plot(cumulative_sample, running_chi2, 'k-', lw=1.5, label='Running mean')
ax.axhline(np.average(chi2, weights=weights), color='red', ls='--', lw=1,
           label=f'Weighted mean = {np.average(chi2, weights=weights):.1f}')
ax.set_xlabel('Sample number')
ax.set_ylabel(r'$\chi^2$')
ax.set_title(r'$\chi^2$ Trace')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/fig_chi2_trace.png')
plt.savefig(f'{outdir}/fig_chi2_trace.pdf')
plt.close()
print("Saved: fig_chi2_trace")

# ============================================================
# Acceptance rate and effective sample size
# ============================================================
total_weight = np.sum(weights)
n_eff_global = total_weight**2 / np.sum(weights**2)
print(f"\nChain statistics:")
print(f"  Total samples: {n}")
print(f"  Total weight: {total_weight:.0f}")
print(f"  Mean weight: {np.mean(weights):.1f}")
print(f"  Global N_eff: {n_eff_global:.0f}")
print(f"  Acceptance rate proxy: {n/total_weight*100:.1f}%")
print(f"  Mean chi2: {np.average(chi2, weights=weights):.1f}")
print(f"  Min chi2: {np.min(chi2):.1f}")

print("\n[DONE]")
