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

# Interplot form factors
q_grid = np.linspace(dq, q_max, N_q)
E_grid = np.linspace(dE, E_max, N_E)
ff_hgte = RegularGridInterpolator((q_grid, E_grid), ff_hgte_grid)
ff_cdte = RegularGridInterpolator((q_grid, E_grid), ff_cdte_grid)
ff_full = RegularGridInterpolator((q_grid, E_grid), ff_full_grid)

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
    integral  = 0.0
    qe_int_grid = np.linspace(0.1, 100, 1000) * nu.eV
    dqe = qe_int_grid[1] - qe_int_grid[0]
    for target in ("hg", "cd", "te"):
        prefactor = weight[target] * rho_DM / mDM / M_cell * nu.aEM * sigma_n * nu.mElectron**2 * (f_p * Z[target] + f_n * (A[target] - Z[target]))**2 / nu.Reduced_Mass(nu.mProton, mDM)**2
        for qi_e in qe_int_grid:
            mN = A[target] * nu.AMU
            qi = qi_e * mN / nu.mElectron
            En = qi**2 / (2.0 * mN)
            dqi = A[target] * nu.AMU / nu.mElectron * dqe
            vMin = v_min(qi, Ee + En, mDM)
            if qi_e <= q_grid[0]:
                ff_qiE = ff_full((q_grid[0], Ee)) * qi_e**2 / q_grid[0]**2   # Dipole approx
            else:
                ff_qiE = ff_full((qi_e, Ee))
            
            integral += prefactor * dqi * qi / qi_e**3 * EtaFunction(vMin) * F_DM(qi)**2 * ff_qiE

    return integral

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
    
    np.savetxt('../data/binned_signal_halo_migdal/binned_signals_Halo_'+ str(j) + '.txt', nq, header=str(m))
    return 0

if __name__ == "__main__":
    arr = np.linspace(0, 8, 9, dtype=int)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compute, arr)