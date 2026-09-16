import numpy as np
from pathlib import Path
import natural_units as nu
from scipy.special import erf
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
# Read form factor files
ff_hgte_grid = np.zeros((N_q,N_E))
ff_cdte_grid = np.zeros((N_q,N_E))
hgte_data_file = np.loadtxt('../data/form_factors/C.HgTe137.dat')
cdte_data_file = np.loadtxt('../data/form_factors/C.CdTe137.dat')
lin2_data_file = np.loadtxt('../data/form_factors/Lin2_HgCdTe.txt')
i=0
for Ei in range(N_E):
    for qi in range(N_q):
        ff_hgte_grid[qi, Ei] = hgte_data_file[i]
        ff_cdte_grid[qi, Ei] = cdte_data_file[i]
        i += 1

# correct by Lindhard:
ff_hgte_grid = ff_hgte_grid / lin2_data_file
ff_cdte_grid = ff_cdte_grid / lin2_data_file

# weights
w_hg, w_cd, w_te = 0.7, 0.3, 1.0
# solid state VCA (virtual crystal approximation):
ff_full_grid = w_hg * ff_hgte_grid + w_cd * ff_cdte_grid

# Interpolate form factors
q_grid = np.linspace(dq, q_max, N_q)
E_grid = np.linspace(dE, E_max, N_E)
ff_full = RegularGridInterpolator((q_grid, E_grid), ff_full_grid)

# 1908.10881: anchor the ionization form factor near q_ref and use its
# dipole scaling below q_ref.  The crystal form factor itself scales as q^5.
q_ref = 0.5 * nu.aEM * nu.mElectron
q_ref_points = (
    np.abs(q_grid - q_ref)
    <= 0.1 * nu.aEM * nu.mElectron + 1.0e-12 * q_ref
)
ff_ion2_ref = np.mean(
    8.0 * nu.aEM * nu.mElectron**2 * E_grid[None, :]
    / q_grid[q_ref_points, None]**3
    * ff_full_grid[q_ref_points],
    axis=0,
)

#atomic masses
weight = {"hg": 0.7, "cd": 0.3, "te": 1.0}
Z = {"hg": 80, "cd": 48, "te": 52}
A = {"hg": 200.59, "cd": 112.41, "te": 127.60}

f_p, f_n = 1, 0

# Halo DM parameters, just in case
rho_DM  = 0.3 * nu.GeV / nu.cm**3
# frac_DM = 1.0   # Adjust later, fix it to be 1 for now.
vesc    = 544.0 * nu.km / nu.sec
v0      = 238.0 * nu.km / nu.sec
v_Earth = 250.2 * nu.km / nu.sec
v_max   = vesc + v_Earth

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
def dRdEe_halo(Ee, sigma_n, mDM):
    integral = 0.0
    mu_n = nu.Reduced_Mass(nu.mProton, mDM)

    for target in ("hg", "cd", "te"):
        mN = A[target] * nu.AMU
        mu_N = nu.Reduced_Mass(mN, mDM)
        discriminant = 1.0 - 2.0 * Ee / (mu_N * v_max**2)
        if discriminant <= 0.0:
            continue

        root = np.sqrt(discriminant)
        q_N_min = 2.0 * Ee / (v_max * (1.0 + root))
        q_N_max = mu_N * v_max * (1.0 + root)
        q_e = np.geomspace(
            q_N_min * nu.mElectron / mN,
            q_N_max * nu.mElectron / mN,
            1000,
        )
        q_N = q_e * mN / nu.mElectron
        E_N = q_N**2 / (2.0 * mN)
        eta = np.array([EtaFunction(v) for v in v_min(q_N, Ee + E_N, mDM)])

        f_ion2 = np.empty_like(q_e)
        low_q = q_e <= q_ref
        f_ion2[low_q] = np.interp(Ee, E_grid, ff_ion2_ref) * (q_e[low_q] / q_ref)**2
        if np.any(~low_q):
            points = np.column_stack((q_e[~low_q], np.full(np.count_nonzero(~low_q), Ee)))
            f_ion2[~low_q] = (
                8.0 * nu.aEM * nu.mElectron**2 * Ee / q_e[~low_q]**3
                * ff_full(points)
            )

        coupling = f_p * Z[target] + f_n * (A[target] - Z[target])
        prefactor = (
            weight[target] * rho_DM / mDM / M_cell
            * sigma_n * coupling**2 / (8.0 * mu_n**2 * Ee)
        )
        integrand = (
            prefactor * (mN / nu.mElectron) * q_N
            * eta * F_DM(q_N)**2 * f_ion2
        )
        integral += np.trapezoid(integrand, q_e)

    return integral

Q_bins = np.arange(1, 11)
Q_edges = energy_gap + epsilon * np.arange(11)
rate_E_grid = np.unique(np.concatenate((
    Q_edges,
    E_grid[(E_grid > Q_edges[0]) & (E_grid < Q_edges[-1])],
)))

# Electron spectrum per mass:
def R_Q_halo(Q, spectrum):
    points = (rate_E_grid >= Q_edges[Q - 1]) & (rate_E_grid <= Q_edges[Q])
    return np.trapezoid(spectrum[points], rate_E_grid[points])

# JWST parameters
pixel_mass = 1.2e-8 *nu.gram
exposure_time   = 3574.278 *nu.sec * 244 / 245  # since we are using (last frame - first frame)
exposure = pixel_mass * exposure_time

log_m_min  = 0
log_m_max  = 4
n_m    = 9   # From GeV  to 10 TeV
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
    out_dir = Path('../data/binned_signal_halo_migdal')
    out_dir.mkdir(parents=True, exist_ok=True)

    spectrum = np.array([dRdEe_halo(Ee, cs, m) for Ee in rate_E_grid])
    nq = np.array([exposure * R_Q_halo(Q, spectrum) for Q in Q_bins])

    np.savetxt(out_dir / ('binned_signals_Halo_' + str(j) + '.txt'), nq, header=str(m))
    return 0

if __name__ == "__main__":
    arr = np.arange(n_m, dtype=int)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(compute, arr))
