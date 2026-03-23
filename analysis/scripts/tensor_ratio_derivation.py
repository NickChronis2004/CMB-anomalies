"""
Tensor-to-scalar ratio prediction from CCR framework
=====================================================
Key derivation:
  r = P_T / P_S  where P_T = 2H_inf^2 / (pi^2 M_Pl^2)  and P_S = A_s

From CCR: H_inf = (1/l_P) sqrt(pi/ln D) = m_Pl sqrt(pi/ln D)

Since M_Pl (reduced) = m_Pl / sqrt(8 pi):
  H_inf / M_Pl = m_Pl sqrt(pi/ln D) * sqrt(8 pi)/m_Pl = sqrt(8 pi^2 / ln D)

r = 2 (H_inf/M_Pl)^2 / (pi^2 A_s)
  = 2 * 8 pi^2 / (ln D * pi^2 * A_s)
  = 16 / (A_s * ln D)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs('./figures', exist_ok=True)

# Parameters
A_s = 2.1e-9  # Planck best-fit scalar amplitude

# ln D range
lnD_values = np.logspace(12, 16, 500)

# r prediction
r_values = 16.0 / (A_s * lnD_values)

# Specific values for table
lnD_table = [1e12, 1e13, 5e13, 1e14, 2e14, 5e14, 1e15, 1e16]
r_table = [16.0/(A_s * x) for x in lnD_table]

print("="*70)
print("CCR Tensor-to-Scalar Ratio: r = 16 / (A_s × ln D)")
print("="*70)
print(f"{'ln D':>12s}  {'r':>12s}  {'Status':>30s}")
print("-"*60)

# Observational bounds
r_planck_BK15 = 0.056  # Planck+BK15 95% CL
r_BICEP3 = 0.036       # BICEP/Keck 2021 approx
r_LiteBIRD = 0.001     # LiteBIRD target sensitivity

for lnD, r in zip(lnD_table, r_table):
    if r > r_planck_BK15:
        status = "EXCLUDED by Planck+BK15"
    elif r > r_BICEP3:
        status = "In tension with BICEP/Keck"
    elif r > r_LiteBIRD:
        status = "Detectable by LiteBIRD"
    else:
        status = "Below LiteBIRD sensitivity"
    print(f"{lnD:>12.1e}  {r:>12.3e}  {status:>30s}")

# Find critical ln D values
lnD_planck = 16.0 / (A_s * r_planck_BK15)
lnD_BICEP = 16.0 / (A_s * r_BICEP3)
lnD_LiteBIRD = 16.0 / (A_s * r_LiteBIRD)

print(f"\nCritical ln D values:")
print(f"  Planck+BK15 (r<0.056):  ln D > {lnD_planck:.2e}")
print(f"  BICEP/Keck (r<0.036):   ln D > {lnD_BICEP:.2e}")
print(f"  LiteBIRD (r~0.001):     ln D > {lnD_LiteBIRD:.2e}")

# Fiducial value
lnD_fid = 2e14
r_fid = 16.0 / (A_s * lnD_fid)
print(f"\nFiducial (ln D = 2×10^14): r = {r_fid:.2e}")

# ---- PLOT ----
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Main curve
ax.plot(lnD_values, r_values, 'b-', linewidth=2.5, label=r'CCR: $r = 16/(A_s \ln D)$')

# Observational bounds
ax.axhline(y=r_planck_BK15, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
           label=r'Planck+BK15 95\% CL: $r < 0.056$')
ax.axhline(y=r_BICEP3, color='orange', linestyle='--', linewidth=1.5, alpha=0.8,
           label=r'BICEP/Keck 2021: $r < 0.036$')
ax.axhline(y=r_LiteBIRD, color='green', linestyle=':', linewidth=1.5, alpha=0.8,
           label=r'LiteBIRD target: $r \sim 10^{-3}$')

# Shade excluded region
ax.fill_between(lnD_values, r_planck_BK15, 1, alpha=0.1, color='red', label='Excluded')

# Fiducial point
ax.plot(lnD_fid, r_fid, 'k*', markersize=15, zorder=5, label=f'Fiducial ($\\ln D = 2\\times 10^{{14}}$)')

# Mark posterior range from MCMC
lnD_post_low = 10**12.8  # ~68% CL lower edge from paper
lnD_post_high = 10**15.2  # ~68% CL upper edge
ax.axvspan(lnD_post_low, lnD_post_high, alpha=0.08, color='blue', label='MCMC 68\% CL range')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ln D$', fontsize=14)
ax.set_ylabel(r'Tensor-to-scalar ratio $r$', fontsize=14)
ax.set_title(r'CCR Prediction for the Tensor-to-Scalar Ratio', fontsize=15)
ax.set_xlim(1e12, 1e16)
ax.set_ylim(1e-7, 1)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('./figures/tensor_ratio_prediction.png', dpi=200, bbox_inches='tight')
plt.savefig('./figures/tensor_ratio_prediction.pdf', bbox_inches='tight')
print("\nPlot saved.")