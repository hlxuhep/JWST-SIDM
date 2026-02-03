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
a_c   = 6.47 * nu.Angstrom
n_val = 32
ne_c  = n_val / a_c**3
kF    = np.power(3 * np.pi**2 * ne_c, 1/3)
vF    = kF / nu.mElectron
omega_p = np.sqrt(4 * np.pi * nu.aEM * ne_c / nu.mElectron)

def lindhard_epsilon(Ee, q, Gamma=None):
    """
    Ee, q 可以是标量或 array（自然单位）。
    ne: 价电子数密度（上一个问题我们算的那个 n_e）
    Gamma: plasmon 宽度，论文里 Γ = 0.1 Ee
    """
    Ee = np.asarray(Ee, dtype=float)
    q  = np.asarray(q,  dtype=float)

    if Gamma is None:
        Gamma = 0.1 * Ee
    Gamma = np.asarray(Gamma, dtype=float)

    # 先保证后面都是复数运算
    q_c   = q.astype(complex)
    Ee_c  = Ee.astype(complex)
    Gam_c = Gamma.astype(complex)

    # u1, u2 按 eq.(4)
    u_common = (Ee_c + 1j * Gam_c) / (q_c * vF)
    u1 = q_c / (2.0 * kF) + u_common
    u2 = q_c / (2.0 * kF) - u_common

    def f(u):
        # 这里的 log 是复数 log
        return 0.5 + kF/(4.0*q_c) * (1.0 - u**2) * np.log((u + 1.0) / (u - 1.0))

    f1 = f(u1)
    f2 = f(u2)

    eps = 1.0 + 3.0 * omega_p**2 / (q_c**2 * vF**2) * (f1 + f2)
    return eps  # 一般后面要用 Im[-1/eps]

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
mA = 0.0

def F_DM(q):
    return ((nu.aEM*nu.mElectron)**2 + mA**2)/(q**2 + mA**2)

def v_min(q, Ee, mDM):
    return Ee/q + q/2/mDM

# Energy spectrum per mass:
def dRdEe_halo(Ee, sigma_n, mDM):
    """Differential Migdal rate dR/dEe per target mass (halo DM).

    Ee : energy deposited into electronic excitations ("omega" in the paper)
    sigma_n : DM-nucleon reference cross section
    mDM : DM mass

    Implements
        dP/domega = (2 alpha v_N^2 / (3 pi^2 omega^4)) * \int dk k^2 Z_ion^2 Im[-1/eps(k,omega)]
    and
        dR/dEe = sum_T \int dE_N (dR_el^T/dE_N) * dP^T(E_N)/dEe ,
    with dR_el/dE_N the usual elastic nuclear recoil rate.
    """
    # Precompute the k-integral I(Ee) = \int dk k^2 Im[-1/eps(k, Ee)]
    # using the same momentum grid q_grid (interpreted here as k).
    eps_k = lindhard_epsilon(Ee, q_grid)
    Im_minus_inv_eps = np.imag(-1.0 / eps_k)  # Im[-1/eps]
    I_k = np.trapz(q_grid**2 * Im_minus_inv_eps, q_grid)

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

        # Maximum recoil energy for this target (from kinematics)
        E_N_max = 2.0 * mu_N**2 * v_max**2 / mN

        # Recoil energy grid for the nucleus
        N_EN = 400
        E_N_grid = np.linspace(0.0, E_N_max, N_EN)
        dE_N = E_N_grid[1] - E_N_grid[0]

        # Corresponding momentum transfer q_N and nuclear velocity v_N
        q_N = np.sqrt(2.0 * mN * E_N_grid)
        v_N_sq = 2.0 * E_N_grid / mN

        # dP/dEe(E_N) using the soft-limit expression
        dP_dEe = pref_P_common * (Z_ion[T]**2) * v_N_sq

        # v_min including the electronic energy cost Ee
        vMin = v_min(q_N, E_N_grid + Ee, mDM)

        # Halo integral eta(v_min)
        eta_vals = np.array([EtaFunction(v) for v in vMin])

        # dR_el/dE_N for this target (standard SI scattering, F_N=1)
        pref_el = (rho_DM / mDM) * N_T * mN * sigma_n / (2.0 * mu_n**2) * Z_eff**2
        integrand = pref_el * F_DM(q_N)**2 * eta_vals * dP_dEe

        # Avoid 0 * inf at E_N = 0 (q_N = 0) which leads to NaN numerically:
        # in the continuum limit the integrand -> 0 at this endpoint, so we set it explicitly.
        integrand[0] = 0.0

        rate += np.trapz(integrand, E_N_grid)

    return rate

# Charge Yield:
def charge_yield(Ee, Q):
    if Ee < energy_gap:
        return 0
    else:
        Ee_1 = epsilon * (Q - 1) + energy_gap
        Ee_2 = epsilon * Q + energy_gap
        if Ee < Ee_1 or Ee > Ee_2:
            return 0.0
        else:
            return 1.0

# Electron spectrum per mass:
def R_Q_halo(Q, sigma_n, mDM):
    R_Q = 0
    for Ei in E_grid:
        cy = charge_yield(Ei, Q)
        R_Q += dE * cy * dRdEe_halo(Ei, sigma_n, mDM)
    return R_Q

# JWST parameters
pixel_mass = 1.2e-8 *nu.gram
exposure_time   = 3574.278 *nu.sec * 244 / 245  # since we are using (last frame - first frame)
exposure = pixel_mass * exposure_time

log_m_min  = -3
log_m_max  = 1
n_m    = 9   # From 1e-3  to 10 GeV
m_grid      = np.logspace(log_m_min, log_m_max, n_m) * nu.GeV

cs_test = 1e-26 * nu.cm * nu.cm

# center_line = np.array([2.15504637e-23, 1.53030461e-23, 1.26359147e-23, 1.61669130e-23,
#       2.40008514e-23, 4.03532201e-23, 6.95747264e-23, 1.40957345e-22,
#       2.84518575e-22, 5.17381239e-22, 1.08351297e-21, 2.15203017e-21,
#       4.12016417e-21, 7.78952222e-21, 1.45615810e-20, 2.70914228e-20,
#       4.94000000e-20]) * nu.cm * nu.cm

def compute(j):
    m = m_grid[j]
    cs = cs_test
    nq = np.zeros(10)
    for q in range(10):
        print('calculating: i=' + str(j) + ' q=' + str(q) +'\n')
        nq[q] = exposure * R_Q_halo(q+1, cs, m)
    
    np.savetxt('../data/binned_signal_halo_migdal_lindhard/binned_signals_Halo_'+ str(j) + '.txt', nq, header=str(m))

    # Make a step plot of the binned signal and save (no display)
    out_dir = Path('../data/binned_signal_halo_migdal_lindhard')
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_bins = np.arange(1, 11)
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
    # Mass index array for parallel computation
    arr = np.arange(n_m, dtype=int)

    # Use a process pool to compute binned signals for each mass point
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compute, arr)