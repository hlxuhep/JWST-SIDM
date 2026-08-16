import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import natural_units as nu
from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
# multi-core/thread:
import concurrent.futures
# HgCdTe form factors' parameters
N_q   = 800
N_E   = 300
dE    = 0.05 * nu.eV
dq    = 0.01 * nu.aEM * nu.mElectron
E_max = N_E * dE
q_max = N_q * dq
energy_gap = 0.234 * nu.eV
epsilon    = 3 * energy_gap
M_cell     = 301.74 * nu.AMU
Q_max      = np.floor((E_max - energy_gap + epsilon) / epsilon)

# q and E grids
q_grid = np.linspace(dq, q_max, N_q)
E_grid = np.linspace(dE, E_max, N_E)

#atomic masses
weight = {"hg": 0.7, "cd": 0.3, "te": 1.0}
Z = {"hg": 80, "cd": 48, "te": 52}
A = {"hg": 200.59, "cd": 112.41, "te": 127.60}

f_p, f_n = 1, 0

# Lindhard Dielectric Function
a_c = 6.47 * nu.Angstrom
ne = 32 / a_c**3
kF = np.power(3 * np.pi**2 * ne, 1/3)
vF = kF / nu.mElectron
omega_p = np.sqrt(4 * np.pi * nu.aEM * ne / nu.mElectron)


def lindhard_epsilon(Ee, q, Gamma=None):
    Ee = np.asarray(Ee, dtype=float)
    q  = np.asarray(q,  dtype=float)

    if Gamma is None:
        Gamma = 0.1 * Ee
    Gamma = np.asarray(Gamma, dtype=float)

    # 先保证后面都是复数运算
    q_c   = q.astype(complex)
    Ee_c  = Ee.astype(complex)
    Gam_c = Gamma.astype(complex)

    u_common = (Ee_c + 1j * Gam_c) / (q_c * vF)
    u1 = q_c / (2.0 * kF) + u_common
    u2 = q_c / (2.0 * kF) - u_common

    def f(u):
        return kF/(4.0*q_c) * (1.0 - u**2) * np.log((u + 1.0) / (u - 1.0))

    bracket = 0.5 + f(u1) + f(u2)
    return 1.0 + 3.0 * omega_p**2 / (q_c**2 * vF**2) * bracket


def loss_integral(Ee, Gamma=None):
    """Return I(Ee) = integral dk k^2 Im[-1/epsilon(k, Ee)]."""
    Ee = np.asarray(Ee, dtype=float)
    Gamma = None if Gamma is None else np.asarray(Gamma, dtype=float)[..., None]
    eps_k = lindhard_epsilon(Ee[..., None], q_grid, Gamma)
    return np.trapezoid(q_grid**2 * np.imag(-1.0 / eps_k), q_grid, axis=-1)

# Halo DM parameters, just in case
rho_DM  = 0.3 * nu.GeV / nu.cm**3
# frac_DM = 1.0   # Adjust later, fix it to be 1 for now.
vesc    = 544.0 * nu.km / nu.sec
v0      = 238.0 * nu.km / nu.sec
v_Earth = 250.2 * nu.km / nu.sec

# DM speed distribution
Nesc = np.pi * v0 * v0 * (np.sqrt(np.pi) * v0 * erf(vesc / v0) - 2 * vesc * np.exp(-vesc * vesc / v0 / v0))
def f_halo(v, vEarth = v_Earth):
	return np.pi * v * v0 * v0 / Nesc / vEarth * (2 * np.exp(-(v * v + vEarth * vEarth) / v0 / v0) * np.sinh(2 * v * vEarth / v0 / v0) + (np.exp(-np.power(v + vEarth, 2.0) / v0 / v0) - np.exp(-vesc * vesc / v0 / v0)) * np.heaviside(np.abs(v + vEarth) - vesc, 0) - (np.exp(-np.power(v - vEarth, 2.0) / v0 / v0) - np.exp(-vesc * vesc / v0 / v0)) * np.heaviside(np.abs(v - vEarth) - vesc, 0))

def EtaFunction(vMin, vE = v_Earth):
    xMin = vMin / v0
    xEsc = vesc / v0
    xE = vE / v0
    eta = 0.0
    if xMin > xE + xEsc:
        eta = 0.0
    elif np.abs(xMin - xE - xEsc) < 1e-8:
        eta = 0.0
    elif xMin > np.abs(xE - xEsc):
        eta = np.power(np.pi, 1.5) * v0 * v0 / 2.0 / Nesc / xE * (erf(xEsc) - erf(xMin - xE) - 2.0 / np.sqrt(np.pi) * (xE + xEsc - xMin) * np.exp(-xEsc * xEsc))
    elif xEsc > xE:
        eta = np.power(np.pi, 1.5) * v0 * v0 / 2.0 / Nesc / xE * (erf(xMin + xE) - erf(xMin - xE) - 4.0 / np.sqrt(np.pi) * xE * np.exp(-xEsc * xEsc))
    else:
        eta = 1.0 / v0 / xE
    return eta

# To determine light or heavy mediator, just in case
mA = 1.0 * nu.TeV
# mA = 0.0

def F_DM(q):
    return ((nu.aEM*nu.mElectron)**2 + mA**2)/(q**2 + mA**2)

def v_min(q, Ee, mDM):
    return Ee/q + q/2/mDM

# Energy spectrum per mass:
def dRdEe_halo(Ee, sigma_n, mDM, I_k=None):
    r"""Differential Migdal rate dR/dEe per target mass (halo DM).

    Ee : energy deposited into electronic excitations ("omega" in the paper)
    sigma_n : DM-nucleon reference cross section
    mDM : DM mass

    Implements
        dP/domega = (2 alpha v_N^2 / (3 pi^2 omega^4)) * \int dk k^2 Z_ion^2 Im[-1/eps(k,omega)]
    and
        dR/dEe = sum_T \int dE_N (dR_el^T/dE_N) * dP^T(E_N)/dEe ,
    with dR_el/dE_N the usual elastic nuclear recoil rate.
    """
    if I_k is None:
        I_k = loss_integral(Ee)

    # Common prefactor from dP/domega
    pref_P_common = 2.0 * nu.aEM * I_k / (3.0 * np.pi**2 * Ee**4)

    # Valence ionic charges for Hg, Cd, Te (Z_ion)
    Z_ion = {"hg": 2.0, "cd": 2.0, "te": 6.0}

    # Reduced mass for DM–nucleon (enters sigma_n normalization)
    mu_n = nu.Reduced_Mass(nu.mProton, mDM)

    # Maximum DM speed in the lab
    v_max = vesc + v_Earth

    rate = 0.0

    for T in ("hg", "cd", "te"):
        mN = A[T] * nu.AMU
        ZN = Z[T]

        # Number of target nuclei of this species per unit detector mass
        N_T = weight[T] / M_cell

        # Coupling combination [f_p Z + f_n (A-Z)]^2
        Z_eff = (f_p * ZN + f_n * (A[T] - ZN))

        # DM–nucleus reduced mass
        mu_N = mN * mDM / (mN + mDM)

        discriminant = 1.0 - 2.0 * Ee / (mu_N * v_max**2)
        if discriminant <= 0.0:
            continue

        root = np.sqrt(discriminant)
        q_N_min = 2.0 * Ee / (v_max * (1.0 + root))
        q_N_max = mu_N * v_max * (1.0 + root)
        q_N = np.geomspace(q_N_min, q_N_max, 200)
        log_q_N = np.log(q_N)
        E_N_grid = q_N**2 / (2.0 * mN)
        v_N_sq = (q_N / mN) ** 2

        # dP/dEe(E_N) using the soft-limit expression
        dP_dEe = pref_P_common * (Z_ion[T]**2) * v_N_sq

        # v_min including the electronic energy cost Ee
        vMin = v_min(q_N, E_N_grid + Ee, mDM)

        # Halo integral eta(v_min)
        eta_vals = np.array([EtaFunction(v) for v in vMin])

        # dR_el/dE_N for this target (standard SI scattering, F_N=1)
        pref_el = (rho_DM / mDM) * N_T * mN * sigma_n / (2.0 * mu_n**2) * Z_eff**2
        integrand = pref_el * F_DM(q_N)**2 * eta_vals * dP_dEe

        rate += np.trapezoid(integrand * q_N**2 / mN, log_q_N)

    return rate

# --- Diagnostics: dP/dEe vs Ee at fixed nuclear momentum transfer q_N ---
Z_ion = {"hg": 2.0, "cd": 2.0, "te": 6.0}

def dP_dEe_fixed_qN(Ee, qN, T, Gamma=None, I_k=None):
    r"""Soft-limit Migdal excitation probability density dP/dEe at fixed q_N.

    Ee : electronic excitation energy (array or scalar)
    qN : nuclear momentum transfer (scalar)
    T  : target species key: 'hg', 'cd', or 'te'

    Implements
        dP/dEe = (2 alpha / (3 pi^2 Ee^4)) * (Z_ion^2 v_N^2) * \int dk k^2 Im[-1/eps(k,Ee)]
    with v_N = qN / mN.
    """
    Ee = np.asarray(Ee, dtype=float)

    if I_k is None:
        I_k = loss_integral(Ee, Gamma)

    pref = 2.0 * nu.aEM * I_k / (3.0 * np.pi**2 * Ee**4)

    mN = A[T] * nu.AMU
    vN_sq = (qN / mN) ** 2

    return pref * (Z_ion[T] ** 2) * vN_sq

def plot_dP_dEe_vs_Ee_fixed_qN(mDM_plot=1.0 * nu.GeV):
    """Save dP/dEe(Ee) curves for q_N = (0.1..1.0) * mDM_plot * v0."""
    out_dir = Path('../data/binned_signal_halo_migdal_lindhard')
    out_dir.mkdir(parents=True, exist_ok=True)

    # q_N scan: 0.1, 0.2, ..., 1.0 times mDM * v0
    factors = np.arange(0.1, 1.01, 0.1)
    qN_list = factors * mDM_plot * v0

    # Ee grid for plotting (start at the band gap to avoid the unphysical Ee->0 divergence)
    Ee_plot = np.linspace(energy_gap, E_max, 600)
    I_k_plot = loss_integral(Ee_plot)

    for T in ("hg", "cd", "te"):
        fig, ax = plt.subplots()
        for fac, qN in zip(factors, qN_list):
            dP = dP_dEe_fixed_qN(Ee_plot, qN, T, I_k=I_k_plot)
            ax.plot(Ee_plot / nu.eV, dP * nu.eV, label=rf"$q_N={fac:.1f}\, m_\chi v_0$")

        ax.set_xlabel(r"$E_e\ \mathrm{[eV]}$")
        ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}E_e\ \mathrm{[1/eV]}$")
        ax.set_title(rf"$\mathrm{{d}}P/\mathrm{{d}}E_e$ vs $E_e$ (fixed $q_N$) — {T.upper()}")
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        fig.tight_layout()

        fig.savefig(out_dir / f"dP_dEe_vs_Ee_fixed_qN_{T}.png")
        plt.close(fig)

    # Also save a combined HgCdTe-cell style curve using mass-fraction weights
    fig, ax = plt.subplots()
    for fac, qN in zip(factors, qN_list):
        dP_hg = dP_dEe_fixed_qN(Ee_plot, qN, "hg", I_k=I_k_plot)
        dP_cd = dP_dEe_fixed_qN(Ee_plot, qN, "cd", I_k=I_k_plot)
        dP_te = dP_dEe_fixed_qN(Ee_plot, qN, "te", I_k=I_k_plot)
        dP_mix = weight["hg"] * dP_hg + weight["cd"] * dP_cd + weight["te"] * dP_te
        ax.plot(Ee_plot / nu.eV, dP_mix * nu.eV, label=rf"$q_N={fac:.1f}\, m_\chi v_0$")

    ax.set_xlabel(r"$E_e\ \mathrm{[eV]}$")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}E_e\ \mathrm{[1/eV]}$")
    ax.set_title(r"$\mathrm{d}P/\mathrm{d}E_e$ vs $E_e$ (fixed $q_N$) — HgCdTe mix")
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dP_dEe_vs_Ee_fixed_qN_mix.png")
    plt.close(fig)

Q_bins = np.arange(1, 11)
Q_edges = energy_gap + epsilon * np.arange(11)
rate_E_grid = np.unique(np.concatenate((
    Q_edges,
    E_grid[(E_grid > Q_edges[0]) & (E_grid < Q_edges[-1])],
)))


def R_Q_halo(Q, spectrum):
    points = (rate_E_grid >= Q_edges[Q - 1]) & (rate_E_grid <= Q_edges[Q])
    return np.trapezoid(spectrum[points], rate_E_grid[points])

# JWST parameters
pixel_mass = 1.2e-8 *nu.gram
exposure_time   = 3574.278 *nu.sec * 244 / 245  # since we are using (last frame - first frame)
exposure = pixel_mass * exposure_time

log_m_min  = -3
log_m_max  = 1
n_m    = 9   # From 1e-3  to 10 GeV
m_grid      = np.logspace(log_m_min, log_m_max, n_m) * nu.GeV

cs_test = 1e-24 * nu.cm * nu.cm

# center_line = np.array([2.15504637e-23, 1.53030461e-23, 1.26359147e-23, 1.61669130e-23,
#       2.40008514e-23, 4.03532201e-23, 6.95747264e-23, 1.40957345e-22,
#       2.84518575e-22, 5.17381239e-22, 1.08351297e-21, 2.15203017e-21,
#       4.12016417e-21, 7.78952222e-21, 1.45615810e-20, 2.70914228e-20,
#       4.94000000e-20]) * nu.cm * nu.cm

def compute(j):
    m = m_grid[j]
    cs = cs_test
    out_dir = Path('../data/binned_signal_halo_migdal_lindhard')
    out_dir.mkdir(parents=True, exist_ok=True)

    I_k_grid = loss_integral(rate_E_grid)
    spectrum = np.array([
        dRdEe_halo(Ee, cs, m, I_k)
        for Ee, I_k in zip(rate_E_grid, I_k_grid)
    ])
    nq = np.array([
        exposure * R_Q_halo(Q, spectrum)
        for Q in Q_bins
    ])

    np.savetxt(out_dir / ('binned_signals_Halo_' + str(j) + '.txt'), nq, header=str(m))

    # Make a step plot of the binned signal and save (no display)
    fig, ax = plt.subplots()
    ax.step(Q_bins, nq, where='mid')
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$N_Q$")
    ax.set_title(rf"$m_\chi = {m/nu.GeV:.2e}\ \mathrm{{GeV}}$")
    fig.tight_layout()

    fig.savefig(out_dir / f"binned_signals_Halo_{j}.png")
    plt.close(fig)
    return 0

if __name__ == "__main__":
    # Diagnostic plots: dP/dEe vs Ee at fixed q_N values
    plot_dP_dEe_vs_Ee_fixed_qN(mDM_plot=1.0 * nu.GeV)
    # Mass index array for parallel computation
    arr = np.arange(n_m, dtype=int)

    # Use a process pool to compute binned signals for each mass point
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compute, arr)
