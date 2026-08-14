"""Silicon Migdal spectra with the calculation written in physical order."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "migdal-si-matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

import natural_units as nu


# Files
here = Path(__file__).resolve().parent
result_dir = here / "result"
csi_file = here.parent / "data/form_factors/C.Si137.dat"
qedark_file = here.parent / "data/QEdark/Si_f2.txt"


# Calculation choices
sigma_n = 1.0e-38 * nu.cm**2
m_chi_100MeV = 0.1 * nu.GeV
m_chi_1GeV = 1.0 * nu.GeV
dE_eV = 0.1
E_max_eV = 60.0
n_q_integral = 1000
n_k_lindhard = 800
n_recoil = 400
n_threads = 32

E_grid_eV = np.arange(dE_eV, E_max_eV + 0.5 * dE_eV, dE_eV)
E_grid = E_grid_eV * nu.eV
dE = E_grid[1] - E_grid[0]


# Essig et al., arXiv:1908.10881v2, Fig. 2, upper-right panel.
# These are the vector-extracted upper edges of the two filled Migdal regions.
paper_E_eV = np.arange(1.7, 50.0, 1.0)
paper_100MeV = np.array((
    0.003706, 0.024086, 0.074220, 0.496003, 1.431343, 2.547316, 3.886451,
    3.501797, 3.172855, 3.699413, 3.314738, 1.715328, 1.029657, 1.399774,
    1.220263, 1.268286, 0.958487, 1.006948, 0.716637, 0.607818, 0.532832,
    0.288063, 0.326775, 0.370849, 0.307732, 0.385278, 0.341534, 0.255247,
    0.211805, 0.319705, 0.370849, 0.311049, 0.311049, 0.316296, 0.277156,
    0.272675, 0.289674, 0.349087, 0.349087, 0.188726, 0.252525, 0.248336,
    0.341534, 0.208381, 0.297735, 0.366737, 0.309320, 0.242963, 0.283406,
))
paper_1GeV = np.array((
    0.082833, 0.541587, 1.687598, 11.399622, 33.066236, 59.506846, 92.321236,
    83.613140, 76.182494, 90.323782, 81.804095, 42.569103, 25.828341,
    35.704758, 31.299899, 32.882387, 25.129022, 26.695552, 19.203832,
    16.470441, 14.594147, 7.978450, 9.152150, 10.498511, 8.805607,
    11.152981, 9.993263, 7.552256, 6.334441, 9.668630, 11.336239, 9.614872,
    9.722689, 10.049136, 8.854840, 8.854840, 9.512330, 11.651716, 11.716864,
    6.405473, 8.663259, 8.663259, 12.042934, 7.468507, 10.790675, 13.440517,
    11.463359, 9.152150, 10.790675,
))


# Silicon
Z = 14.0
A = 28.0855
m_N = A * nu.AMU
M_cell = 2.0 * m_N
Z_ion = 4.0
q_0 = 0.5 * nu.aEM * nu.mElectron


# Standard halo model
rho_DM = 0.3 * nu.GeV / nu.cm**3
v_esc = 544.0 * nu.km / nu.sec
v_0 = 238.0 * nu.km / nu.sec
v_Earth = 250.2 * nu.km / nu.sec
v_max = v_esc + v_Earth

N_esc = np.pi * v_0**2 * (np.sqrt(np.pi) * v_0 * erf(v_esc / v_0) - 2.0 * v_esc * np.exp(-(v_esc / v_0)**2))


# Silicon free-electron gas used by the dielectric function
a_c = 5.431 * nu.Angstrom
n_val = 32.0
n_e = n_val / a_c**3
k_F = (3.0 * np.pi**2 * n_e) ** (1.0 / 3.0)
v_F = k_F / nu.mElectron
omega_p = np.sqrt(4.0 * np.pi * nu.aEM * n_e / nu.mElectron)


def lindhard_epsilon(omega, q):
    """Damped Lindhard dielectric function with Gamma = 0.1 omega."""
    omega = np.asarray(omega, dtype=complex)
    q = np.asarray(q, dtype=complex)
    gamma = 0.1 * omega

    u_common = (omega + 1j * gamma) / (q * v_F)
    u1 = q / (2.0 * k_F) + u_common
    u2 = q / (2.0 * k_F) - u_common

    f1 = k_F / (4.0 * q) * (1.0 - u1**2) * np.log((u1 + 1.0) / (u1 - 1.0))
    f2 = k_F / (4.0 * q) * (1.0 - u2**2) * np.log((u2 + 1.0) / (u2 - 1.0))

    return 1.0 + 3.0 * omega_p**2 / (q**2 * v_F**2) * (0.5 + f1 + f2)


# -----------------------------------------------------------------------------
# Crystal form-factor calculation: C.Si137, QEDark, screened QEDark
# -----------------------------------------------------------------------------

# Both tables use dq = 0.02 alpha m_e and dE = 0.1 eV.
dq_ff = 0.02 * nu.aEM * nu.mElectron
dE_ff = 0.1 * nu.eV

# C.Si137: 1250 q points x 500 energy points, stored energy-major.
csi_nq = 1250
csi_ne = 500
csi_q = np.arange(1, csi_nq + 1) * dq_ff
csi_e = np.arange(1, csi_ne + 1) * dE_ff
csi_f2 = np.loadtxt(csi_file).reshape(csi_ne, csi_nq).T
csi_interp = RegularGridInterpolator((csi_q, csi_e), csi_f2, bounds_error=False, fill_value=0.0)

# QEDark: 900 q points x 500 energy points, with the dimensions in line one.
qedark_nq = 900
qedark_ne = 500
qedark_q = np.arange(1, qedark_nq + 1) * dq_ff
qedark_e = np.arange(1, qedark_ne + 1) * dE_ff
qedark_f2 = np.loadtxt(qedark_file, skiprows=1).reshape(qedark_ne, qedark_nq).T
qedark_interp = RegularGridInterpolator((qedark_q, qedark_e), qedark_f2, bounds_error=False, fill_value=0.0)

# Screened QEDark table.
qedark_epsilon = lindhard_epsilon(qedark_e[None, :], qedark_q[:, None])
qedark_scr_f2 = qedark_f2 / np.abs(qedark_epsilon)**2
qedark_scr_interp = RegularGridInterpolator((qedark_q, qedark_e), qedark_scr_f2, bounds_error=False, fill_value=0.0)

# Average the tables near q_0 = 0.5 alpha m_e.
csi_q0_points = np.abs(csi_q - q_0) <= 0.1 * nu.aEM * nu.mElectron + 1.0e-12 * q_0
qedark_q0_points = np.abs(qedark_q - q_0) <= 0.04 * nu.aEM * nu.mElectron + 1.0e-12 * q_0

csi_f2_q0 = np.mean(csi_f2[csi_q0_points], axis=0)
qedark_f2_q0 = np.mean(qedark_f2[qedark_q0_points], axis=0)
qedark_scr_f2_q0 = np.mean(qedark_scr_f2[qedark_q0_points], axis=0)

csi_fion_q0 = 8.0 * nu.aEM * nu.mElectron**2 * csi_e / q_0**3 * csi_f2_q0
qedark_fion_q0 = 8.0 * nu.aEM * nu.mElectron**2 * qedark_e / q_0**3 * qedark_f2_q0
qedark_scr_fion_q0 = 8.0 * nu.aEM * nu.mElectron**2 * qedark_e / q_0**3 * qedark_scr_f2_q0


def mean_inverse_speed(v_min):
    """Mean inverse speed for the truncated Maxwell distribution."""
    x_min = np.asarray(v_min, dtype=float) / v_0
    x_E = v_Earth / v_0
    x_esc = v_esc / v_0
    eta = np.zeros_like(x_min)

    allowed = x_min < x_E + x_esc
    upper = allowed & (x_min > abs(x_E - x_esc))
    lower = allowed & ~upper

    eta[upper] = np.pi**1.5 * v_0**2 / (2.0 * N_esc * x_E) * (
        erf(x_esc)
        - erf(x_min[upper] - x_E)
        - 2.0 / np.sqrt(np.pi)
        * (x_E + x_esc - x_min[upper])
        * np.exp(-x_esc**2)
    )

    eta[lower] = np.pi**1.5 * v_0**2 / (2.0 * N_esc * x_E) * (
        erf(x_min[lower] + x_E)
        - erf(x_min[lower] - x_E)
        - 4.0 / np.sqrt(np.pi) * x_E * np.exp(-x_esc**2)
    )

    return eta


def electron_momentum_grid(omega, m_chi, q_table_max):
    """Allowed q_e = (m_e/m_N) q_N interval for a fixed omega."""
    mu_chi_N = nu.Reduced_Mass(m_N, m_chi)
    discriminant = 1.0 - 2.0 * omega / (mu_chi_N * v_max**2)
    if discriminant <= 0.0:
        return np.array([])

    root = np.sqrt(discriminant)
    q_N_min = 2.0 * omega / (v_max * (1.0 + root))
    q_N_max = mu_chi_N * v_max * (1.0 + root)
    q_e_min = q_N_min * nu.mElectron / m_N
    q_e_max = min(q_N_max * nu.mElectron / m_N, q_table_max)

    if q_e_max <= q_e_min:
        return np.array([])
    return np.linspace(q_e_min, q_e_max, n_q_integral)


def ionization_form_factor(q_e, omega, E_table, fion_q0, interp):
    """Crystal-to-ionization conversion and low-q dipole scaling."""
    f_ion2 = np.zeros_like(q_e)

    low_q = q_e <= q_0
    high_q = q_e > q_0

    f_ion2_q0 = np.interp(omega, E_table, fion_q0)
    f_ion2[low_q] = f_ion2_q0 * (q_e[low_q] / q_0)**2

    interpolation_points = np.column_stack((q_e[high_q], np.full(np.count_nonzero(high_q), omega)))
    f_crystal2 = interp(interpolation_points)
    f_ion2[high_q] = 8.0 * nu.aEM * nu.mElectron**2 * omega / q_e[high_q]**3 * f_crystal2

    return f_ion2


def crystal_migdal_rate(omega, m_chi, q_table, E_table, fion_q0, interp):
    """dR/domega from one tabulated Silicon crystal form factor."""
    if omega < E_table[0] or omega > E_table[-1]:
        return 0.0

    q_e = electron_momentum_grid(omega, m_chi, q_table[-1])
    if q_e.size == 0:
        return 0.0

    q_N = q_e * m_N / nu.mElectron
    E_N = q_N**2 / (2.0 * m_N)
    v_min = (omega + E_N) / q_N + q_N / (2.0 * m_chi)

    f_ion2 = ionization_form_factor(q_e, omega, E_table, fion_q0, interp)

    sigma_p = sigma_n * (A / Z) ** 2
    mu_chi_p = nu.Reduced_Mass(nu.mProton, m_chi)
    prefactor = rho_DM / m_chi / M_cell * sigma_p * Z**2 / (8.0 * mu_chi_p**2)

    integrand = (
        prefactor * (m_N / nu.mElectron) * q_N / omega
        * mean_inverse_speed(v_min) * f_ion2
    )
    return float(trapezoid(integrand, q_e))


# -----------------------------------------------------------------------------
# Lindhard energy-loss calculation
# -----------------------------------------------------------------------------

lindhard_k = np.linspace(0.01, 8.0, n_k_lindhard) * nu.aEM * nu.mElectron


def lindhard_loss_integral(omega):
    """Integral dk k^2 Im[-1/epsilon(k, omega)]."""
    epsilon = lindhard_epsilon(omega, lindhard_k)
    loss_function = np.imag(-1.0 / epsilon)
    return float(trapezoid(lindhard_k**2 * loss_function, lindhard_k))


def minimum_speed(q, deposited_energy, m_chi):
    """Minimum incoming DM speed, with q = 0 assigned infinity."""
    q = np.asarray(q, dtype=float)
    v_min = np.full_like(q, np.inf)
    nonzero = q > 0.0
    v_min[nonzero] = deposited_energy[nonzero] / q[nonzero] + q[nonzero] / (2.0 * m_chi)
    return v_min


def lindhard_migdal_rate(omega, m_chi, loss_integral):
    """dR/domega from the Lindhard energy-loss function."""
    mu_chi_p = nu.Reduced_Mass(nu.mProton, m_chi)
    mu_chi_N = nu.Reduced_Mass(m_N, m_chi)
    E_N_max = 2.0 * mu_chi_N**2 * v_max**2 / m_N

    E_N = np.linspace(0.0, E_N_max, n_recoil)
    q_N = np.sqrt(2.0 * m_N * E_N)
    v_N2 = 2.0 * E_N / m_N

    dP_domega = 2.0 * nu.aEM * loss_integral / (3.0 * np.pi**2 * omega**4) * Z_ion**2 * v_N2

    sigma_p = sigma_n * (A / Z) ** 2
    N_T = 1.0 / m_N
    elastic_prefactor = rho_DM / m_chi * N_T * m_N * sigma_p * Z**2 / (2.0 * mu_chi_p**2)

    deposited_energy = E_N + omega
    v_min = minimum_speed(q_N, deposited_energy, m_chi)
    integrand = elastic_prefactor * mean_inverse_speed(v_min) * dP_domega
    integrand[0] = 0.0
    return float(trapezoid(integrand, E_N))


def evaluate_on_energy_grid(function):
    """Evaluate one specified spectrum on the 0.1 eV energy grid."""
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        values = pool.map(function, E_grid)
        return np.fromiter(values, dtype=float, count=E_grid.size)


def average_over_one_ev(rate):
    """Return E times the mean rate in each 1 eV energy interval."""
    edges = np.arange(0.0, E_max_eV + 1.0e-9, 1.0) * nu.eV
    centers = 0.5 * (edges[:-1] + edges[1:])
    dR_dlnE = np.zeros_like(centers)

    for index in range(centers.size):
        in_interval = (E_grid >= edges[index]) & (E_grid < edges[index + 1])
        average_rate = np.sum(rate[in_interval]) * dE / (1.0 * nu.eV)
        dR_dlnE[index] = centers[index] * average_rate * nu.kg * nu.year

    return centers, dR_dlnE


def main():
    result_dir.mkdir(parents=True, exist_ok=True)

    # C.Si137 spectra
    csi_100MeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_100MeV, csi_q, csi_e, csi_fion_q0, csi_interp))
    csi_1GeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_1GeV, csi_q, csi_e, csi_fion_q0, csi_interp))

    # QEDark spectra
    qedark_100MeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_100MeV, qedark_q, qedark_e, qedark_fion_q0, qedark_interp))
    qedark_1GeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_1GeV, qedark_q, qedark_e, qedark_fion_q0, qedark_interp))

    # Screened QEDark spectra
    qedark_scr_100MeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_100MeV, qedark_q, qedark_e, qedark_scr_fion_q0, qedark_scr_interp))
    qedark_scr_1GeV = evaluate_on_energy_grid(lambda w: crystal_migdal_rate(w, m_chi_1GeV, qedark_q, qedark_e, qedark_scr_fion_q0, qedark_scr_interp))

    # Lindhard energy-loss spectra
    lindhard_loss = evaluate_on_energy_grid(lindhard_loss_integral)
    lindhard_100MeV = evaluate_on_energy_grid(lambda w: lindhard_migdal_rate(w, m_chi_100MeV, np.interp(w, E_grid, lindhard_loss)))
    lindhard_1GeV = evaluate_on_energy_grid(lambda w: lindhard_migdal_rate(w, m_chi_1GeV, np.interp(w, E_grid, lindhard_loss)))

    # Differential rates on the 0.1 eV grid
    raw_columns = np.column_stack((
        E_grid / nu.eV,
        csi_100MeV * nu.kg * nu.year * nu.eV,
        qedark_100MeV * nu.kg * nu.year * nu.eV,
        qedark_scr_100MeV * nu.kg * nu.year * nu.eV,
        lindhard_100MeV * nu.kg * nu.year * nu.eV,
        csi_1GeV * nu.kg * nu.year * nu.eV,
        qedark_1GeV * nu.kg * nu.year * nu.eV,
        qedark_scr_1GeV * nu.kg * nu.year * nu.eV,
        lindhard_1GeV * nu.kg * nu.year * nu.eV,
    ))
    raw_header = ",".join((
        "E_eV",
        "dRdE_csi137_100MeV_per_kg_yr_eV",
        "dRdE_qedark_100MeV_per_kg_yr_eV",
        "dRdE_qedark_screened_100MeV_per_kg_yr_eV",
        "dRdE_lindhard_100MeV_per_kg_yr_eV",
        "dRdE_csi137_1GeV_per_kg_yr_eV",
        "dRdE_qedark_1GeV_per_kg_yr_eV",
        "dRdE_qedark_screened_1GeV_per_kg_yr_eV",
        "dRdE_lindhard_1GeV_per_kg_yr_eV",
    ))
    raw_output = result_dir / "migdal_Si_dRdE_0p1eV.csv"
    np.savetxt(raw_output, raw_columns, delimiter=",", header=raw_header, comments="")

    # Rates averaged over 1 eV energy intervals
    E_centers, csi_100MeV_1eV = average_over_one_ev(csi_100MeV)
    _, qedark_100MeV_1eV = average_over_one_ev(qedark_100MeV)
    _, qedark_scr_100MeV_1eV = average_over_one_ev(qedark_scr_100MeV)
    _, lindhard_100MeV_1eV = average_over_one_ev(lindhard_100MeV)
    _, csi_1GeV_1eV = average_over_one_ev(csi_1GeV)
    _, qedark_1GeV_1eV = average_over_one_ev(qedark_1GeV)
    _, qedark_scr_1GeV_1eV = average_over_one_ev(qedark_scr_1GeV)
    _, lindhard_1GeV_1eV = average_over_one_ev(lindhard_1GeV)

    spectra_columns = np.column_stack((
        E_centers / nu.eV,
        csi_100MeV_1eV,
        qedark_100MeV_1eV,
        qedark_scr_100MeV_1eV,
        lindhard_100MeV_1eV,
        csi_1GeV_1eV,
        qedark_1GeV_1eV,
        qedark_scr_1GeV_1eV,
        lindhard_1GeV_1eV,
    ))
    spectra_header = raw_header.replace("E_eV", "E_center_eV").replace("dRdE_", "dRdlnE_").replace("_per_kg_yr_eV", "_per_kg_yr")
    spectra_output = result_dir / "migdal_Si_dRdlnE_1eV.csv"
    np.savetxt(spectra_output, spectra_columns, delimiter=",", header=spectra_header, comments="")

    # Overlay figure
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)

    axes[0].plot(E_centers / nu.eV, csi_100MeV_1eV, lw=2.0, color="#1f77b4", label="C.Si137")
    axes[0].plot(E_centers / nu.eV, qedark_100MeV_1eV, lw=2.0, color="#ff7f0e", label="QEDark")
    axes[0].plot(E_centers / nu.eV, qedark_scr_100MeV_1eV, lw=2.0, color="#9467bd", label=r"QEDark $/|\epsilon|^2$")
    axes[0].plot(E_centers / nu.eV, lindhard_100MeV_1eV, lw=2.0, color="#2ca02c", label="Lindhard")
    axes[0].plot(paper_E_eV, paper_100MeV, lw=2.2, ls="--", color="black", label="1908.10881 Fig. 2")

    axes[1].plot(E_centers / nu.eV, csi_1GeV_1eV, lw=2.0, color="#1f77b4")
    axes[1].plot(E_centers / nu.eV, qedark_1GeV_1eV, lw=2.0, color="#ff7f0e")
    axes[1].plot(E_centers / nu.eV, qedark_scr_1GeV_1eV, lw=2.0, color="#9467bd")
    axes[1].plot(E_centers / nu.eV, lindhard_1GeV_1eV, lw=2.0, color="#2ca02c")
    axes[1].plot(paper_E_eV, paper_1GeV, lw=2.2, ls="--", color="black")

    axes[0].set_title(r"$m_\chi=100\,\mathrm{MeV}$")
    axes[1].set_title(r"$m_\chi=1\,\mathrm{GeV}$")
    axes[0].set_ylabel(r"$\mathrm{d}R/\mathrm{d}\ln\Delta E\ [1/(\mathrm{kg\,yr})]$")
    axes[0].legend(frameon=False)

    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlim(0.0, 60.0)
        axis.set_ylim(2.0e-5, 2.0e3)
        axis.set_xlabel(r"$\Delta E\ [\mathrm{eV}]$")
        axis.grid(alpha=0.2, which="both")

    fig.suptitle(r"Si Migdal, $\bar{\sigma}_n=10^{-38}\ \mathrm{cm}^2$, $F_{\rm DM}=1$")
    fig.tight_layout()
    figure_output = result_dir / "migdal_Si_overlay.png"
    fig.savefig(figure_output, dpi=220)
    plt.close(fig)

    print(f"saved: {raw_output}")
    print(f"saved: {spectra_output}")
    print(f"saved: {figure_output}")


if __name__ == "__main__":
    main()
