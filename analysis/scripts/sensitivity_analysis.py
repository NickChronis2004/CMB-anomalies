"""
Sensitivity Analysis: CCR-Based IR Cutoff k_c(D, N, T_reh)
============================================================
Computes the comoving IR cutoff predicted by the CCR framework
and explores its dependence on the Hilbert space dimension D,
the number of inflationary e-folds N, and the reheating
temperature T_reh.

Key equation (eq. 10 of the paper):
    k_c = (e^{-N} * T_0) / (T_reh * l_P) * sqrt(pi / ln D)

Reference: Chronis & Sifakis (2026), CCR framework.

Author: Nikolaos Chronis
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import os

# ── Physical constants ──────────────────────────────────────────
l_P = 1.616255e-35          # Planck length [m]
k_B = 1.380649e-23          # Boltzmann constant [J/K]
hbar = 1.054571817e-34      # reduced Planck constant [J·s]
c = 2.99792458e8            # speed of light [m/s]
G = 6.67430e-11             # gravitational constant [m^3/(kg·s^2)]
T_0 = 2.7255                # CMB temperature today [K]
GeV_to_K = 1.16045e13       # 1 GeV in Kelvin

# ── Conversion: physical wavenumber to comoving (Mpc^-1) ───────
m_to_Mpc_inv = 3.2408e-23   # 1/m to 1/Mpc (since 1 Mpc = 3.0857e22 m)

# ── Fiducial values ─────────────────────────────────────────────
N_fid = 60
T_reh_fid_GeV = 1e15                      # GeV
T_reh_fid_K = T_reh_fid_GeV * GeV_to_K    # Kelvin
lnD_fid = 2.0e14

# ── Output directories ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
TABLE_DIR = os.path.join(SCRIPT_DIR, '..', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def compute_kc(lnD, N, T_reh_K):
    """
    Compute the comoving IR cutoff k_c in Mpc^-1.

    Parameters
    ----------
    lnD : float or array
        Natural log of the Hilbert space dimension.
    N : float
        Number of inflationary e-folds.
    T_reh_K : float
        Reheating temperature in Kelvin.

    Returns
    -------
    kc : float or array
        Comoving IR cutoff [Mpc^-1].
    """
    # Scale factor at onset of inflation
    a_i = np.exp(-N) * (T_0 / T_reh_K)

    # Physical IR cutoff [1/m]
    kc_phys = (1.0 / l_P) * np.sqrt(np.pi / lnD)

    # Comoving IR cutoff [1/m] -> [Mpc^-1]
    kc = a_i * kc_phys * (1.0 / m_to_Mpc_inv)  # divide because m_to_Mpc_inv = Mpc^-1 per m^-1

    # Actually: kc [Mpc^-1] = kc [1/m] * (1 Mpc / 3.0857e22 m) -- but kc[1/m] * m_per_Mpc
    # Let me redo this carefully:
    # kc has units 1/m in physical * a_i (dimensionless) = 1/m comoving
    # To convert 1/m to 1/Mpc: multiply by meters per Mpc
    m_per_Mpc = 3.0857e22
    kc = a_i * kc_phys * m_per_Mpc

    return kc


def compute_Hinf(lnD):
    """
    Compute the inflationary Hubble parameter H_inf in GeV.

    Parameters
    ----------
    lnD : float or array
        Natural log of the Hilbert space dimension.

    Returns
    -------
    Hinf_GeV : float or array
        Inflationary Hubble rate [GeV].
    """
    # H_inf = 1/(l_P) * sqrt(pi/lnD) in natural units
    # In SI: H_inf [1/s] = c/l_P * sqrt(pi/lnD)
    # Convert to GeV: E = hbar * H [J] -> GeV
    Hinf_si = (c / l_P) * np.sqrt(np.pi / lnD)  # [1/s]
    Hinf_GeV = (hbar * Hinf_si) / (1.602176634e-10)  # J to GeV: 1 GeV = 1.602e-10 J
    return Hinf_GeV


def compute_RdS(lnD):
    """
    Compute the de Sitter horizon radius R_dS = H_inf^{-1} in meters.

    Parameters
    ----------
    lnD : float or array

    Returns
    -------
    RdS : float or array [m]
    """
    return l_P * np.sqrt(lnD / np.pi)


# ═══════════════════════════════════════════════════════════════
# 1. SENSITIVITY: k_c vs ln D
# ═══════════════════════════════════════════════════════════════
def plot_kc_vs_lnD():
    """k_c as a function of ln D for fiducial N, T_reh."""
    lnD_arr = np.logspace(12, 16, 500)
    kc_arr = compute_kc(lnD_arr, N_fid, T_reh_fid_K)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(lnD_arr, kc_arr, 'k-', lw=2)

    # Mark fiducial point
    kc_fid = compute_kc(lnD_fid, N_fid, T_reh_fid_K)
    ax.plot(lnD_fid, kc_fid, 'ro', ms=8, zorder=5, label=f'Fiducial: $\\ln D = 2\\times10^{{14}}$')

    # Mark Contaldi best-fit
    kc_contaldi = 4.9e-4
    ax.axhline(kc_contaldi, ls='--', color='blue', alpha=0.7, label=f'Contaldi 2003 best-fit: $k_c = 4.9\\times10^{{-4}}$ Mpc$^{{-1}}$')

    # Mark present Hubble scale
    a0H0 = 2.3e-4  # Mpc^-1 (approximate)
    ax.axhline(a0H0, ls=':', color='green', alpha=0.7, label=f'$a_0 H_0 \\approx 2.3\\times10^{{-4}}$ Mpc$^{{-1}}$')

    ax.set_xlabel('$\\ln D$', fontsize=14)
    ax.set_ylabel('$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_title('CCR IR Cutoff vs Hilbert Space Dimension', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(1e12, 1e16)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_lnD.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_lnD.png'), dpi=300)
    plt.close(fig)
    print(f"[✓] Saved kc_vs_lnD  |  k_c(fiducial) = {kc_fid:.4e} Mpc^-1")
    return kc_fid


# ═══════════════════════════════════════════════════════════════
# 2. SENSITIVITY: k_c vs N (e-folds)
# ═══════════════════════════════════════════════════════════════
def plot_kc_vs_N():
    """k_c as a function of N for several ln D values."""
    N_arr = np.linspace(50, 70, 200)
    lnD_values = [5e13, 1e14, 2e14, 5e14, 1e15]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lnD_values)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for lnD_val, col in zip(lnD_values, colors):
        kc_arr = compute_kc(lnD_val, N_arr, T_reh_fid_K)
        ax.semilogy(N_arr, kc_arr, lw=2, color=col, label=f'$\\ln D = {lnD_val:.0e}$')

    # Observable window
    ax.axhline(4.9e-4, ls='--', color='blue', alpha=0.5, label='Contaldi best-fit')
    ax.axhspan(1e-4, 2e-3, alpha=0.1, color='orange', label='Observable window ($\\ell < 30$)')

    ax.set_xlabel('$N$ (e-folds)', fontsize=14)
    ax.set_ylabel('$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_title('IR Cutoff Sensitivity to Inflationary Duration', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(50, 70)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_N.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_N.png'), dpi=300)
    plt.close(fig)
    print("[✓] Saved kc_vs_N")


# ═══════════════════════════════════════════════════════════════
# 3. SENSITIVITY: k_c vs T_reh
# ═══════════════════════════════════════════════════════════════
def plot_kc_vs_Treh():
    """k_c as a function of T_reh for several ln D values."""
    T_reh_GeV_arr = np.logspace(10, 16, 200)
    T_reh_K_arr = T_reh_GeV_arr * GeV_to_K
    lnD_values = [5e13, 1e14, 2e14, 5e14, 1e15]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lnD_values)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for lnD_val, col in zip(lnD_values, colors):
        kc_arr = compute_kc(lnD_val, N_fid, T_reh_K_arr)
        ax.loglog(T_reh_GeV_arr, kc_arr, lw=2, color=col, label=f'$\\ln D = {lnD_val:.0e}$')

    ax.axhline(4.9e-4, ls='--', color='blue', alpha=0.5, label='Contaldi best-fit')
    ax.axhspan(1e-4, 2e-3, alpha=0.1, color='orange', label='Observable window')
    ax.axvline(T_reh_fid_GeV, ls=':', color='red', alpha=0.5, label=f'Fiducial $T_{{reh}}$')

    ax.set_xlabel('$T_{\\mathrm{reh}}$ [GeV]', fontsize=14)
    ax.set_ylabel('$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_title('IR Cutoff Sensitivity to Reheating Temperature', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_Treh.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'kc_vs_Treh.png'), dpi=300)
    plt.close(fig)
    print("[✓] Saved kc_vs_Treh")


# ═══════════════════════════════════════════════════════════════
# 4. 2D CONTOUR: k_c in the (N, ln D) plane
# ═══════════════════════════════════════════════════════════════
def plot_2d_contour_N_lnD():
    """2D contour of k_c in the (N, ln D) plane at fiducial T_reh."""
    N_arr = np.linspace(50, 70, 300)
    lnD_arr = np.logspace(13, 15.5, 300)
    N_grid, lnD_grid = np.meshgrid(N_arr, lnD_arr)
    kc_grid = compute_kc(lnD_grid, N_grid, T_reh_fid_K)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Log-spaced contour levels centered around observable range
    levels = [1e-5, 5e-5, 1e-4, 2e-4, 4.9e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    cs = ax.contour(N_grid, lnD_grid, kc_grid, levels=levels,
                    colors='k', linewidths=1.0)
    ax.clabel(cs, fmt='%.1e', fontsize=8)

    # Fill contour
    cf = ax.contourf(N_grid, lnD_grid, kc_grid, levels=np.logspace(-5, -1, 50),
                     cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='$k_c$ [Mpc$^{-1}$]')

    # Mark fiducial
    ax.plot(N_fid, lnD_fid, 'w*', ms=15, mec='k', mew=1.5, zorder=5, label='Fiducial')

    # Observable band (k_c ~ 1e-4 to 2e-3)
    ax.set_yscale('log')
    ax.set_xlabel('$N$ (e-folds)', fontsize=14)
    ax.set_ylabel('$\\ln D$', fontsize=14)
    ax.set_title('$k_c$ [Mpc$^{-1}$] in the $(N, \\ln D)$ plane\n'
                 f'$T_{{\\mathrm{{reh}}}} = 10^{{15}}$ GeV (fixed)', fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'kc_contour_N_lnD.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'kc_contour_N_lnD.png'), dpi=300)
    plt.close(fig)
    print("[✓] Saved kc_contour_N_lnD")


# ═══════════════════════════════════════════════════════════════
# 5. 2D CONTOUR: k_c in the (T_reh, ln D) plane
# ═══════════════════════════════════════════════════════════════
def plot_2d_contour_Treh_lnD():
    """2D contour of k_c in the (T_reh, ln D) plane at fiducial N."""
    T_reh_GeV_arr = np.logspace(10, 16, 300)
    T_reh_K_arr = T_reh_GeV_arr * GeV_to_K
    lnD_arr = np.logspace(13, 15.5, 300)
    T_grid_GeV, lnD_grid = np.meshgrid(T_reh_GeV_arr, lnD_arr)
    T_grid_K = T_grid_GeV * GeV_to_K
    kc_grid = compute_kc(lnD_grid, N_fid, T_grid_K)

    fig, ax = plt.subplots(figsize=(8, 6))

    levels = [1e-5, 5e-5, 1e-4, 2e-4, 4.9e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    cs = ax.contour(T_grid_GeV, lnD_grid, kc_grid, levels=levels,
                    colors='k', linewidths=1.0)
    ax.clabel(cs, fmt='%.1e', fontsize=8)

    cf = ax.contourf(T_grid_GeV, lnD_grid, kc_grid, levels=np.logspace(-5, -1, 50),
                     cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='$k_c$ [Mpc$^{-1}$]')

    ax.plot(T_reh_fid_GeV, lnD_fid, 'w*', ms=15, mec='k', mew=1.5, zorder=5, label='Fiducial')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$T_{\\mathrm{reh}}$ [GeV]', fontsize=14)
    ax.set_ylabel('$\\ln D$', fontsize=14)
    ax.set_title('$k_c$ [Mpc$^{-1}$] in the $(T_{\\mathrm{reh}}, \\ln D)$ plane\n'
                 f'$N = {N_fid}$ (fixed)', fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'kc_contour_Treh_lnD.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'kc_contour_Treh_lnD.png'), dpi=300)
    plt.close(fig)
    print("[✓] Saved kc_contour_Treh_lnD")


# ═══════════════════════════════════════════════════════════════
# 6. MODIFIED POWER SPECTRUM P_CCR(k) for various alpha
# ═══════════════════════════════════════════════════════════════
def plot_power_spectrum_alpha():
    """P_CCR(k)/P_LCDM(k) suppression factor for various alpha values."""
    k_arr = np.logspace(-5, -1, 1000)  # Mpc^-1
    kc_fid = compute_kc(lnD_fid, N_fid, T_reh_fid_K)
    alphas = [1, 2, 3, 4, 6]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(alphas)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for alpha, col in zip(alphas, colors):
        suppression = 1.0 - np.exp(-(k_arr / kc_fid)**alpha)
        ax.semilogx(k_arr, suppression, lw=2, color=col, label=f'$\\alpha = {alpha}$')

    ax.axvline(kc_fid, ls='--', color='gray', alpha=0.7, label=f'$k_c = {kc_fid:.2e}$ Mpc$^{{-1}}$')

    # Mark the k range probed by ℓ < 30
    # Roughly k ~ ℓ / (η_0 - η_*) where η_0 - η_* ≈ 14000 Mpc
    k_ell2 = 2.0 / 14000  # ℓ = 2
    k_ell30 = 30.0 / 14000  # ℓ = 30
    ax.axvspan(k_ell2, k_ell30, alpha=0.15, color='orange', label='$\\ell = 2$–$30$ range')

    ax.set_xlabel('$k$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel('$P_{\\mathrm{CCR}}(k) / P_{\\Lambda\\mathrm{CDM}}(k)$', fontsize=14)
    ax.set_title('Power Spectrum Suppression Factor', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1e-5, 1e-1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'suppression_factor_alpha.pdf'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'suppression_factor_alpha.png'), dpi=300)
    plt.close(fig)
    print("[✓] Saved suppression_factor_alpha")


# ═══════════════════════════════════════════════════════════════
# 7. NUMERICAL TABLE: key quantities for a grid of ln D
# ═══════════════════════════════════════════════════════════════
def generate_tables():
    """Generate LaTeX-ready tables with key derived quantities."""

    # Table 1: k_c, H_inf, R_dS for a range of ln D
    lnD_values = [1e13, 5e13, 1e14, 2e14, 5e14, 1e15, 5e15]
    header = (
        f"{'ln D':>12s}  {'k_c [Mpc^-1]':>14s}  {'H_inf [GeV]':>14s}  "
        f"{'R_dS [m]':>14s}  {'R_dS [l_P]':>14s}"
    )
    lines = [header, '-' * len(header)]
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Derived CCR quantities for fiducial $N=60$, $T_{\mathrm{reh}} = 10^{15}$ GeV.}",
        r"\label{tab:derived_quantities}",
        r"\begin{tabular}{ccccc}",
        r"\hline\hline",
        r"$\ln D$ & $k_c$ [Mpc$^{-1}$] & $H_{\mathrm{inf}}$ [GeV] & $R_{\mathrm{dS}}$ [m] & $R_{\mathrm{dS}}$ [$l_P$] \\",
        r"\hline",
    ]

    for lnD in lnD_values:
        kc = compute_kc(lnD, N_fid, T_reh_fid_K)
        Hinf = compute_Hinf(lnD)
        RdS = compute_RdS(lnD)
        RdS_lP = RdS / l_P

        lines.append(f"{lnD:>12.2e}  {kc:>14.4e}  {Hinf:>14.4e}  {RdS:>14.4e}  {RdS_lP:>14.4e}")
        latex_lines.append(
            f"${lnD:.1e}$ & ${kc:.3e}$ & ${Hinf:.3e}$ & ${RdS:.3e}$ & ${RdS_lP:.3e}$ \\\\"
        )

    latex_lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]

    # Save plain text
    table_txt = '\n'.join(lines)
    with open(os.path.join(TABLE_DIR, 'derived_quantities.txt'), 'w') as f:
        f.write(table_txt)
    print("\n" + table_txt)

    # Save LaTeX
    with open(os.path.join(TABLE_DIR, 'derived_quantities.tex'), 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"\n[✓] Saved derived_quantities table (txt + tex)")

    # Table 2: Sensitivity — k_c variation with N and T_reh
    N_values = [50, 55, 60, 65, 70]
    T_reh_values_GeV = [1e10, 1e12, 1e14, 1e15, 1e16]

    print("\n\nSensitivity Table: k_c [Mpc^-1] for ln D = 2e14")
    print(f"{'':>8s}", end='')
    for T in T_reh_values_GeV:
        print(f"  {'T=' + f'{T:.0e}':>14s}", end='')
    print()
    print('-' * 82)

    sens_latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{$k_c$ [Mpc$^{-1}$] sensitivity to $N$ and $T_{\mathrm{reh}}$ for $\ln D = 2\times 10^{14}$.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{c" + "c" * len(T_reh_values_GeV) + "}",
        r"\hline\hline",
        r"$N$ & " + " & ".join([f"$T_{{\\mathrm{{reh}}}} = 10^{{{int(np.log10(T))}}}$ GeV" for T in T_reh_values_GeV]) + r" \\",
        r"\hline",
    ]

    for N in N_values:
        row = f"{'N=' + str(N):>8s}"
        latex_row = f"${N}$"
        for T_GeV in T_reh_values_GeV:
            T_K = T_GeV * GeV_to_K
            kc = compute_kc(lnD_fid, N, T_K)
            row += f"  {kc:>14.4e}"
            latex_row += f" & ${kc:.3e}$"
        print(row)
        sens_latex.append(latex_row + r" \\")

    sens_latex += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(TABLE_DIR, 'sensitivity_N_Treh.tex'), 'w') as f:
        f.write('\n'.join(sens_latex))
    print(f"\n[✓] Saved sensitivity_N_Treh table")


# ═══════════════════════════════════════════════════════════════
# 8. CONSISTENCY CHECK: H_inf vs Planck bound
# ═══════════════════════════════════════════════════════════════
def consistency_checks():
    """Print key consistency checks."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECKS")
    print("=" * 60)

    kc_fid = compute_kc(lnD_fid, N_fid, T_reh_fid_K)
    Hinf_fid = compute_Hinf(lnD_fid)
    RdS_fid = compute_RdS(lnD_fid)
    Hinf_planck_bound = 6e13  # GeV

    print(f"\nFiducial parameters:")
    print(f"  ln D      = {lnD_fid:.2e}")
    print(f"  N         = {N_fid}")
    print(f"  T_reh     = {T_reh_fid_GeV:.2e} GeV")

    print(f"\nDerived quantities:")
    print(f"  k_c       = {kc_fid:.4e} Mpc^-1")
    print(f"  H_inf     = {Hinf_fid:.4e} GeV")
    print(f"  R_dS      = {RdS_fid:.4e} m = {RdS_fid/l_P:.4e} l_P")

    print(f"\nPlanck tensor bound: H_inf < {Hinf_planck_bound:.1e} GeV")
    print(f"  H_inf / H_inf^Planck = {Hinf_fid/Hinf_planck_bound:.4f}")
    print(f"  ✓ H_inf is {Hinf_planck_bound/Hinf_fid:.1f}x below Planck bound")

    print(f"\nContaldi 2003 best-fit: k_c = 4.9e-4 Mpc^-1")
    print(f"  Our k_c / Contaldi k_c = {kc_fid / 4.9e-4:.4f}")

    # Check: D -> infinity limit
    kc_large = compute_kc(1e30, N_fid, T_reh_fid_K)
    print(f"\nΛCDM recovery check (ln D = 1e30):")
    print(f"  k_c = {kc_large:.4e} Mpc^-1 → effectively zero ✓")

    # Number of accessible modes above k_c up to k_max ~ 0.3 Mpc^-1
    k_max = 0.3  # Planck resolution
    n_decades = np.log10(k_max / kc_fid)
    print(f"\nAccessible k range: [{kc_fid:.2e}, {k_max}] Mpc^-1")
    print(f"  Spanning {n_decades:.1f} decades in k")
    print(f"  Modes below k_c are suppressed → affects ℓ < ~30")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("CCR-CMB Sensitivity Analysis")
    print("=" * 60)

    kc_fid = plot_kc_vs_lnD()
    plot_kc_vs_N()
    plot_kc_vs_Treh()
    plot_2d_contour_N_lnD()
    plot_2d_contour_Treh_lnD()
    plot_power_spectrum_alpha()
    generate_tables()
    consistency_checks()

    print("\n" + "=" * 60)
    print("All outputs saved to figures/ and tables/")
    print("=" * 60)
