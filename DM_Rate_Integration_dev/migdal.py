import numpy as np
from pathlib import Path
import re
import natural_units as nu
from scipy.interpolate import RegularGridInterpolator
import concurrent.futures

# HgCdTe form factor 参数
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

module_dir = Path(__file__).resolve().parent
data_dir = module_dir.parent / 'data'
speed_pdf_dir = data_dir / 'DM_Speed_PDF_test'
binned_signal_dir = data_dir / 'binned_signal_shielded_migdal'

# 读取 form factor 文件
ff_hgte_grid = np.zeros((N_q,N_E))
ff_cdte_grid = np.zeros((N_q,N_E))
hgte_data_file = np.loadtxt(data_dir / 'form_factors/C.HgTe137.dat')
cdte_data_file = np.loadtxt(data_dir / 'form_factors/C.CdTe137.dat')
lin2_data_file = np.loadtxt(data_dir / 'form_factors/Lin2_HgCdTe.txt')
i=0
for Ei in range(N_E):
    for qi in range(N_q):
        ff_hgte_grid[qi, Ei] = hgte_data_file[i]
        ff_cdte_grid[qi, Ei] = cdte_data_file[i]
        i += 1

# 加入 Lindhard dielectric function
ff_hgte_grid = ff_hgte_grid / lin2_data_file
ff_cdte_grid = ff_cdte_grid / lin2_data_file

# HgCdTe 中的元素比例
w_hg, w_cd, w_te = 0.7, 0.3, 1.0
ff_full_grid = w_hg * ff_hgte_grid + w_cd * ff_cdte_grid

# 对 form factor 插值
q_grid = np.linspace(dq, q_max, N_q)
E_grid = np.linspace(dE, E_max, N_E)
ff_full = RegularGridInterpolator((q_grid, E_grid), ff_full_grid)

# 按 1908.10881 在 q_ref 附近确定电离 form factor，并在低动量区使用 dipole scaling
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

# 原子质量与化学计量系数
weight = {"hg": 0.7, "cd": 0.3, "te": 1.0}
A = {"hg": 200.59, "cd": 112.41, "te": 127.60}

# Halo DM 密度
rho_DM  = 0.3 * nu.GeV / nu.cm**3

def F_DM(q):
    return np.ones_like(q)

def v_min(q, Ee, mDM):
    return Ee/q + q/2/mDM

def build_eta_function(speed_pdf):
    if speed_pdf.ndim != 2 or speed_pdf.shape[1] != 2:
        raise ValueError('DM speed PDF 必须包含两列数据。')
    if speed_pdf.shape[0] < 3:
        raise ValueError('DM speed PDF 至少需要三个速度点。')
    if not np.all(np.isfinite(speed_pdf)):
        raise ValueError('DM speed PDF 含有非有限数值。')

    speed = speed_pdf[:, 0]
    pdf = speed_pdf[:, 1]
    if not np.isclose(speed[0], 0.0, rtol=0.0, atol=1.0e-16):
        raise ValueError('DM speed PDF 的速度网格必须从零开始。')
    if np.any(np.diff(speed) <= 0.0):
        raise ValueError('DM speed PDF 的速度网格必须严格递增。')
    if np.any(pdf < 0.0):
        raise ValueError('DM speed PDF 不能包含负值。')

    slopes = np.diff(pdf) / np.diff(speed)
    intercepts = pdf[:-1] - slopes * speed[:-1]
    interval_integrals = np.zeros(speed.size - 1)
    positive_intervals = speed[:-1] > 0.0
    lower = speed[:-1][positive_intervals]
    upper = speed[1:][positive_intervals]
    interval_integrals[positive_intervals] = (
        slopes[positive_intervals] * (upper - lower)
        + intercepts[positive_intervals] * np.log(upper / lower)
    )

    eta_at_nodes = np.zeros(speed.size)
    for index in range(speed.size - 2, 0, -1):
        eta_at_nodes[index] = eta_at_nodes[index + 1] + interval_integrals[index]

    def eta_function(v_minimum):
        v_minimum = np.asarray(v_minimum, dtype=float)
        if np.any(v_minimum <= 0.0):
            raise ValueError('v_min 必须为正数。')

        eta = np.zeros_like(v_minimum)
        inside = v_minimum < speed[-1]
        interval_indices = np.searchsorted(
            speed,
            v_minimum[inside],
            side='right',
        ) - 1
        interval_indices = np.clip(interval_indices, 0, speed.size - 2)
        interval_upper = speed[interval_indices + 1]
        eta[inside] = (
            slopes[interval_indices] * (interval_upper - v_minimum[inside])
            + intercepts[interval_indices]
            * np.log(interval_upper / v_minimum[inside])
            + eta_at_nodes[interval_indices + 1]
        )
        return eta

    return eta_function, speed[-1]

# 每个暗物质质量和散射截面的电子能谱
def dRdEe(Ee, sigma_n, mDM, eta_function, v_max):
    integral = 0.0
    mu_n = nu.Reduced_Mass(nu.mNucleon, mDM)

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
        eta = eta_function(v_min(q_N, Ee + E_N, mDM))

        f_ion2 = np.empty_like(q_e)
        low_q = q_e <= q_ref
        f_ion2[low_q] = np.interp(Ee, E_grid, ff_ion2_ref) * (q_e[low_q] / q_ref)**2
        if np.any(~low_q):
            points = np.column_stack((q_e[~low_q], np.full(np.count_nonzero(~low_q), Ee)))
            f_ion2[~low_q] = (
                8.0 * nu.aEM * nu.mElectron**2 * Ee / q_e[~low_q]**3
                * ff_full(points)
            )

        coupling = A[target]
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

N_bins = 20
Q_bins = np.arange(1, N_bins+1)
Q_edges = energy_gap + epsilon * np.arange(N_bins+1)
rate_E_grid = np.unique(np.concatenate((
    Q_edges,
    E_grid[(E_grid > Q_edges[0]) & (E_grid < Q_edges[-1])],
)))

# 对每个 Q 区间积分
def R_Q(Q, spectrum):
    points = (rate_E_grid >= Q_edges[Q - 1]) & (rate_E_grid <= Q_edges[Q])
    return np.trapezoid(spectrum[points], rate_E_grid[points])

# JWST 参数
pixel_mass = 1.2e-8 *nu.gram
exposure_time   = 3574.278 *nu.sec * 244 / 245  # 使用末帧与首帧之间的取数时间
exposure = pixel_mass * exposure_time

# 参数网格与 generate_grid_and_script.ipynb 保持一致
m_grid_GeV = np.array([1.0, 10.0, 100.0, 1.0e3, 1.0e4])
center_line_cm2 = np.array([2.15e-25, 1.00e-26, 1.00e-26, 6.00e-26, 6.00e-25])
cs_scale_grid = np.logspace(-3, 1, 5)
sigma_grid_cm2 = center_line_cm2[:, None] * cs_scale_grid[None, :]
speed_pdf_pattern = re.compile(
    r'DM_Speed_PDF_mDM=([0-9.eE+-]+)_GeV_'
    r'sigma=([0-9.eE+-]+)_cm2_vMin=([0-9.eE+-]+)_kps'
)

def find_grid_index(grid, value, name):
    matches = np.flatnonzero(np.isclose(grid, value, rtol=1.0e-12, atol=0.0))
    if matches.size != 1:
        raise ValueError(f'{name}={value} 与参数网格不符。')
    return int(matches[0])

def discover_jobs():
    jobs = {}
    for speed_pdf_path in sorted(speed_pdf_dir.glob('*.txt')):
        match = speed_pdf_pattern.fullmatch(speed_pdf_path.stem)
        if match is None:
            raise ValueError(f'无法解析文件名：{speed_pdf_path.name}')

        mass_text, sigma_text, v_min_text = match.groups()
        mass_GeV = float(mass_text)
        sigma_cm2 = float(sigma_text)
        if float(v_min_text) != 0.0:
            raise ValueError(f'输入文件要求 vMin=0：{speed_pdf_path.name}')

        mass_index = find_grid_index(m_grid_GeV, mass_GeV, 'mDM')
        cross_section_index = find_grid_index(
            sigma_grid_cm2[mass_index],
            sigma_cm2,
            'sigma_n',
        )
        grid_index = (mass_index, cross_section_index)
        if grid_index in jobs:
            raise ValueError(f'参数点重复：{grid_index}')

        output_path = binned_signal_dir / (
            f'binned_signals_mDM={mass_text}_GeV_'
            f'sigma={sigma_text}_cm2.txt'
        )
        jobs[grid_index] = (
            speed_pdf_path,
            output_path,
            mass_GeV,
            sigma_cm2,
        )

    expected_job_count = m_grid_GeV.size * cs_scale_grid.size
    if len(jobs) != expected_job_count:
        raise RuntimeError(
            f'应有 {expected_job_count} 个 DM speed PDF，实际找到 {len(jobs)} 个。'
        )

    return [
        jobs[(mass_index, cross_section_index)]
        for mass_index in range(m_grid_GeV.size)
        for cross_section_index in range(cs_scale_grid.size)
    ]

def compute(job):
    speed_pdf_path, output_path, mass_GeV, sigma_cm2 = job
    speed_pdf = np.loadtxt(speed_pdf_path)
    eta_function, v_max = build_eta_function(speed_pdf)
    mDM = mass_GeV * nu.GeV
    sigma_n = sigma_cm2 * nu.cm * nu.cm

    spectrum = np.array([
        dRdEe(Ee, sigma_n, mDM, eta_function, v_max)
        for Ee in rate_E_grid
    ])
    nq = np.array([exposure * R_Q(Q, spectrum) for Q in Q_bins])
    if not np.all(np.isfinite(nq)) or np.any(nq < 0.0):
        raise RuntimeError(f'信号计算结果无效：{speed_pdf_path.name}')

    np.savetxt(output_path, nq)
    return output_path.name

if __name__ == "__main__":
    if not speed_pdf_dir.is_dir():
        raise FileNotFoundError(f'DM speed PDF 目录不存在：{speed_pdf_dir}')
    binned_signal_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        completed_files = list(executor.map(compute, jobs))
    print(f'已生成 {len(completed_files)} 个 Migdal binned signal 文件。')
