import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12

# CCR predictions as functions of ln D
lnD = np.logspace(12, 16, 500)
As = 2.1e-9

# Fiducial thermal history
N = 60
T_reh_GeV = 1e15
T_reh_K = T_reh_GeV * 1.16045e13
T0 = 2.7255
l_P = 1.616255e-35  # m
Mpc = 3.0857e22     # m
g_ratio = (3.938 / 106.75)**(1./3.)

kc = np.exp(-N) * g_ratio * (T0 / T_reh_K) * (1./l_P) \
     * np.sqrt(np.pi / lnD) * Mpc  # Mpc^-1
r = 16. / (As * lnD)

fig, ax = plt.subplots(figsize=(8, 6))

# Main CCR curve
ax.plot(kc, r, 'b-', lw=2.5, label=r'CCR: $r = 16/(A_s \ln D)$', zorder=3)

# Mark specific ln D values
for lnD_mark, label, marker in [
    (2e14, r'$\ln D = 2\times10^{14}$ (fiducial)', '*'),
    (2.2e13, r'$\ln D = 2.2\times10^{13}$ (Contaldi)', 'D'),
    (1e13, r'$\ln D = 10^{13}$ ($3\sigma$ threshold)', 's'),
]:
    idx = np.argmin(np.abs(lnD - lnD_mark))
    ax.plot(kc[idx], r[idx], marker=marker, ms=12, color='k', zorder=5,
            label=label)

# Current/future sensitivity
ax.axhline(0.036, color='r', ls='--', alpha=0.7, lw=1.5,
           label=r'BICEP/Keck 2021 ($r < 0.036$)')
ax.axhline(1e-3, color='green', ls=':', alpha=0.7, lw=1.5,
           label=r'LiteBIRD target ($\sigma_r \sim 10^{-3}$)')
ax.axvspan(4e-4, 8e-4, color='purple', alpha=0.08,
           label=r'Contaldi $k_c$ range')

# Observable window
ax.axvspan(1e-4, 2e-3, color='orange', alpha=0.05, zorder=0)
ax.text(3.5e-4, 3e-7, 'Observable\nwindow', fontsize=9,
        color='orange', alpha=0.7, ha='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]', fontsize=14)
ax.set_ylabel(r'$r$', fontsize=14)
ax.set_title(r'CCR consistency relation: $k_c(D)$ vs $r(D)$', fontsize=14)
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.set_xlim(1e-6, 1e-1)
ax.set_ylim(1e-8, 1e-1)
ax.grid(True, alpha=0.2)

plt.tight_layout()
import os
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/consistency_kc_r.pdf', dpi=150)
plt.savefig('../figures/consistency_kc_r.png', dpi=150)
print("Done!")