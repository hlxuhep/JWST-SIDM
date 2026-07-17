"""Si Migdal spectra from C.Si137, QEdark, and a Lindhard estimate.

Full run on the 5950X:
    python migdal_Si.py --jobs 32 --qedark /path/to/QEdark-python/Si_f2.txt

Lightweight check:
    python migdal_Si.py --smoke
"""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

_MPL = os.environ.get("MPLCONFIGDIR")
if not _MPL or not os.access(_MPL, os.W_OK):
    _MPL = str(Path(tempfile.gettempdir()) / "migdal-si-mpl")
    Path(_MPL).mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _MPL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

import natural_units as nu

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


HERE = Path(__file__).resolve().parent
FF_DIR = HERE.parent / "data/form_factors"

# Silicon and halo inputs
Z = 14.0
A = 28.0855
M_N = A * nu.AMU
M_CELL = 2.0 * M_N
Z_ION = 4.0
RHO = 0.3 * nu.GeV / nu.cm**3
V_ESC = 544.0 * nu.km / nu.sec
V_0 = 238.0 * nu.km / nu.sec
V_E = 250.2 * nu.km / nu.sec
V_MAX = V_ESC + V_E
Q0 = 0.5 * nu.aEM * nu.mElectron

# Lindhard inputs copied from migdal_lindhard_Si.py
LATTICE = 5.431 * nu.Angstrom
N_VAL = 32.0
N_ELEC = N_VAL / LATTICE**3
K_F = (3.0 * np.pi**2 * N_ELEC) ** (1.0 / 3.0)
V_F = K_F / nu.mElectron
W_P = np.sqrt(4.0 * np.pi * nu.aEM * N_ELEC / nu.mElectron)

_N_ESC = np.pi * V_0**2 * (
    np.sqrt(np.pi) * V_0 * erf(V_ESC / V_0)
    - 2.0 * V_ESC * np.exp(-(V_ESC / V_0) ** 2)
)


@dataclass
class FF:
    name: str
    q: np.ndarray
    e: np.ndarray
    f2: np.ndarray
    half: float

    def __post_init__(self):
        self.itp = RegularGridInterpolator(
            (self.q, self.e), self.f2, bounds_error=False, fill_value=0.0
        )
        use = np.abs(self.q - Q0) <= self.half + 1.0e-12 * Q0
        if not np.any(use):
            raise ValueError(f"{self.name}: q0 window is outside the table")
        crystal = np.mean(self.f2[use], axis=0)
        self.anchor = (
            8.0 * nu.aEM * nu.mElectron**2 * self.e / Q0**3 * crystal
        )


def qd_path(given=None):
    """Locate the old 900 x 500 QEdark Si table."""
    if given:
        path = Path(given).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(path)

    env = os.environ.get("QEDARK_SI_F2")
    choices = [
        Path(env).expanduser() if env else None,
        FF_DIR / "Si_f2.txt",
        Path.home() / "Documents/2023-1/QEdark-main/QEdark-python/Si_f2.txt",
    ]
    for path in choices:
        if path and path.is_file():
            return path.resolve()
    raise FileNotFoundError("QEdark Si_f2.txt not found; use --qedark PATH")


def load_ff(path, name, nq=None, ne=None, half=0.1):
    """Load an E-major, q-minor QEdark-style table."""
    path = Path(path)
    with path.open() as stream:
        head = stream.readline().split()

    header = len(head) == 2 and all(word.lstrip("+-").isdigit() for word in head)
    if header:
        file_nq, file_ne = map(int, head)
        nq = file_nq if nq is None else nq
        ne = file_ne if ne is None else ne
        if (nq, ne) != (file_nq, file_ne):
            raise ValueError(f"{path}: header says {file_nq} x {file_ne}")
        data = np.loadtxt(path, skiprows=1)
    else:
        if nq is None or ne is None:
            raise ValueError(f"{path}: nq and ne are required without a header")
        data = np.loadtxt(path)

    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size != nq * ne:
        raise ValueError(f"{path}: got {data.size} values, expected {nq * ne}")

    dq = 0.02 * nu.aEM * nu.mElectron
    de = 0.1 * nu.eV
    q = np.arange(1, nq + 1) * dq
    e = np.arange(1, ne + 1) * de
    f2 = data.reshape(ne, nq).T
    return FF(name, q, e, f2, half * nu.aEM * nu.mElectron)


def ion_ff(qe, e, ff):
    """Convert crystal |f|^2 to the ionization form factor used by Migdal."""
    scalar = np.ndim(qe) == 0
    qe = np.atleast_1d(np.asarray(qe, dtype=float))
    out = np.zeros_like(qe)
    if e < ff.e[0] or e > ff.e[-1]:
        return float(out[0]) if scalar else out

    low = (qe >= 0.0) & (qe <= Q0)
    if np.any(low):
        anchor = np.interp(e, ff.e, ff.anchor)
        out[low] = anchor * (qe[low] / Q0) ** 2

    high = (qe > Q0) & (qe <= ff.q[-1])
    if np.any(high):
        points = np.column_stack((qe[high], np.full(np.count_nonzero(high), e)))
        out[high] = (
            8.0
            * nu.aEM
            * nu.mElectron**2
            * e
            / qe[high] ** 3
            * ff.itp(points)
        )
    return float(out[0]) if scalar else out


def eta(vmin):
    """Mean inverse speed for the truncated boosted Maxwell distribution."""
    scalar = np.ndim(vmin) == 0
    x = np.atleast_1d(np.asarray(vmin, dtype=float)) / V_0
    xe = V_E / V_0
    xesc = V_ESC / V_0
    out = np.zeros_like(x)

    allowed = x < xe + xesc
    upper = allowed & (x > abs(xe - xesc))
    out[upper] = np.pi**1.5 * V_0**2 / (2.0 * _N_ESC * xe) * (
        erf(xesc)
        - erf(x[upper] - xe)
        - 2.0 / np.sqrt(np.pi) * (xe + xesc - x[upper]) * np.exp(-xesc**2)
    )

    lower = allowed & ~upper
    if xesc > xe:
        out[lower] = np.pi**1.5 * V_0**2 / (2.0 * _N_ESC * xe) * (
            erf(x[lower] + xe)
            - erf(x[lower] - xe)
            - 4.0 / np.sqrt(np.pi) * xe * np.exp(-xesc**2)
        )
    else:
        out[lower] = 1.0 / (V_0 * xe)
    return float(out[0]) if scalar else out


def vmin(q, e, mchi):
    q = np.asarray(q, dtype=float)
    e = np.broadcast_to(np.asarray(e, dtype=float), q.shape)
    out = np.full_like(q, np.inf)
    use = q > 0.0
    out[use] = e[use] / q[use] + q[use] / (2.0 * mchi)
    return out


def qe_span(e, mchi, qmax, n=1000):
    """Electron-frame q grid allowed by nuclear recoil kinematics."""
    mu = nu.Reduced_Mass(M_N, mchi)
    disc = 1.0 - 2.0 * e / (mu * V_MAX**2)
    if disc <= 0.0:
        return None
    root = np.sqrt(disc)
    q_lo = 2.0 * e / (V_MAX * (1.0 + root))
    q_hi = mu * V_MAX * (1.0 + root)
    qe_lo = q_lo * nu.mElectron / M_N
    qe_hi = min(q_hi * nu.mElectron / M_N, qmax)
    if qe_hi <= qe_lo:
        return None
    return np.linspace(qe_lo, qe_hi, n)


def sigma_p(sigma_n):
    """Paper convention: sigma_n = (Z/A)^2 sigma_p for proton coupling."""
    return sigma_n * (A / Z) ** 2


def rate_tab(e, sigma_n, mchi, ff, nq=1000):
    """dR/dE from a tabulated crystal form factor."""
    if e < ff.e[0] or e > ff.e[-1]:
        return 0.0
    qe = qe_span(e, mchi, ff.q[-1], nq)
    if qe is None:
        return 0.0

    qn = qe * M_N / nu.mElectron
    en = qn**2 / (2.0 * M_N)
    vm = e + en
    vm = vm / qn + qn / (2.0 * mchi)
    pref = (
        RHO
        / mchi
        / M_CELL
        * sigma_p(sigma_n)
        * Z**2
        / (8.0 * nu.Reduced_Mass(nu.mProton, mchi) ** 2)
    )
    fion = ion_ff(qe, e, ff)
    y = pref * (M_N / nu.mElectron) * qn / e * eta(vm) * fion
    return float(trapezoid(y, qe))


def eps_lin(e, q):
    """Damped Lindhard dielectric function."""
    q = np.asarray(q, dtype=complex)
    ec = np.asarray(e, dtype=complex)
    gamma = 0.1 * ec
    common = (ec + 1j * gamma) / (q * V_F)
    u1 = q / (2.0 * K_F) + common
    u2 = q / (2.0 * K_F) - common

    def f(u):
        return 0.5 + K_F / (4.0 * q) * (1.0 - u**2) * np.log((u + 1.0) / (u - 1.0))

    return 1.0 + 3.0 * W_P**2 / (q**2 * V_F**2) * (f(u1) + f(u2))


def loss_lin(e, nk=800):
    """Integral of k^2 Im[-1/epsilon], the Lindhard electronic response."""
    k = np.linspace(0.01, 8.0, nk) * nu.aEM * nu.mElectron
    loss = np.imag(-1.0 / eps_lin(e, k))
    return float(trapezoid(k**2 * loss, k))


def rate_lin(e, sigma_n, mchi, loss, nr=400):
    """dR/dE from the Lindhard loss function."""
    mu_p = nu.Reduced_Mass(nu.mProton, mchi)
    mu_n = nu.Reduced_Mass(M_N, mchi)
    en_max = 2.0 * mu_n**2 * V_MAX**2 / M_N
    en = np.linspace(0.0, en_max, nr)
    qn = np.sqrt(2.0 * M_N * en)
    vn2 = 2.0 * en / M_N

    prob = 2.0 * nu.aEM * loss / (3.0 * np.pi**2 * e**4) * Z_ION**2 * vn2
    pref = RHO / mchi / M_N * M_N * sigma_p(sigma_n) * Z**2 / (2.0 * mu_p**2)
    y = pref * eta(vmin(qn, en + e, mchi)) * prob
    y[0] = 0.0
    return float(trapezoid(y, en))


def work(fn, items, jobs=1, label=None):
    items = list(items)
    if jobs == 1:
        values = map(fn, items)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=jobs)
        values = pool.map(fn, items)
    if tqdm is not None:
        values = tqdm(values, total=len(items), desc=label, leave=False)
    out = np.fromiter(values, dtype=float, count=len(items))
    if pool is not None:
        pool.shutdown()
    return out


def bin_1ev(e, raw, emax):
    edges = np.arange(0.0, emax + 1.0e-9, 1.0) * nu.eV
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for key, values in raw.items():
        binned = np.zeros_like(centers)
        for i, center in enumerate(centers):
            use = (e >= edges[i]) & (e < edges[i + 1])
            if np.any(use):
                de = e[1] - e[0]
                avg = np.sum(values[use]) * de / (1.0 * nu.eV)
                binned[i] = center * avg * nu.kg * nu.year
        out[key] = binned
    return centers, out


def mtag(mchi):
    mev = mchi / nu.MeV
    return f"{mev:g}MeV" if mev < 1000.0 else f"{mchi / nu.GeV:g}GeV"


def mlabel(mchi):
    mev = mchi / nu.MeV
    if mev < 1000.0:
        return rf"{mev:g}\,\mathrm{{MeV}}"
    return rf"{mchi / nu.GeV:g}\,\mathrm{{GeV}}"


def sci(value):
    power = int(np.floor(np.log10(abs(value))))
    lead = value / 10.0**power
    if np.isclose(lead, 1.0):
        return rf"10^{{{power}}}"
    return rf"{lead:g}\times10^{{{power}}}"


def save(e, raw, centers, spectra, masses, stem="migdal_Si"):
    keys = [(model, mass) for mass in masses for model in ("ours", "qedark", "lindhard")]

    raw_cols = [e / nu.eV]
    raw_head = ["E_eV"]
    for key in keys:
        raw_cols.append(raw[key] * nu.kg * nu.year * nu.eV)
        raw_head.append(f"dRdE_{key[0]}_{mtag(key[1])}_per_kg_yr_eV")
    np.savetxt(
        HERE / f"{stem}_raw.csv",
        np.column_stack(raw_cols),
        delimiter=",",
        header=",".join(raw_head),
        comments="",
    )

    spec_cols = [centers / nu.eV]
    spec_head = ["E_center_eV"]
    for key in keys:
        spec_cols.append(spectra[key])
        spec_head.append(f"dRdlnE_{key[0]}_{mtag(key[1])}_per_kg_yr")
    np.savetxt(
        HERE / f"{stem}_spectra.csv",
        np.column_stack(spec_cols),
        delimiter=",",
        header=",".join(spec_head),
        comments="",
    )


def plot_all(centers, spectra, masses, sigma_n, stem="migdal_Si"):
    colors = {"ours": "#1f77b4", "qedark": "#ff7f0e", "lindhard": "#2ca02c"}
    labels = {"ours": "C.Si137", "qedark": "QEdark", "lindhard": "Lindhard"}
    fig, axes = plt.subplots(1, len(masses), figsize=(10.0, 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mass in zip(axes, masses):
        for model in ("ours", "qedark", "lindhard"):
            ax.plot(
                centers / nu.eV,
                spectra[(model, mass)],
                lw=2.0,
                color=colors[model],
                label=labels[model],
            )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 60.0)
        ax.set_ylim(2.0e-5, 2.0e3)
        ax.set_xlabel(r"$\Delta E\ \mathrm{[eV]}$")
        ax.set_title(rf"$m_\chi={mlabel(mass)}$")
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel(r"$\mathrm{d}R/\mathrm{d}\ln\Delta E\ \mathrm{[1/(kg\,yr)]}$")
    axes[0].legend(frameon=False)
    fig.suptitle(
        rf"Si Migdal, $\bar{{\sigma}}_n={sci(sigma_n / nu.cm**2)}\ \mathrm{{cm}}^2$, "
        rf"$F_{{\rm DM}}=1$"
    )
    fig.tight_layout()
    fig.savefig(HERE / f"{stem}_overlay.png", dpi=220)
    plt.close(fig)


def run(qedark=None, jobs=1, de=0.1, emax=60.0, nq=1000, nk=800, nr=400):
    ours = load_ff(FF_DIR / "C.Si137.dat", "C.Si137", 1250, 500, half=0.1)
    qdark = load_ff(qd_path(qedark), "QEdark", half=0.04)
    tables = {"ours": ours, "qedark": qdark}

    sigma_n = 1.0e-38 * nu.cm**2
    masses = (0.1 * nu.GeV, 1.0 * nu.GeV)
    e = np.arange(de, emax + 0.5 * de, de) * nu.eV
    raw = {}

    losses = work(partial(loss_lin, nk=nk), e, jobs, "Lindhard response")
    for mass in masses:
        for model, ff in tables.items():
            fn = partial(rate_tab, sigma_n=sigma_n, mchi=mass, ff=ff, nq=nq)
            raw[(model, mass)] = work(fn, e, jobs, f"{model}, {mtag(mass)}")

        pairs = list(zip(e, losses))
        fn = lambda pair: rate_lin(pair[0], sigma_n, mass, pair[1], nr)
        raw[("lindhard", mass)] = work(fn, pairs, jobs, f"lindhard, {mtag(mass)}")

    centers, spectra = bin_1ev(e, raw, emax)
    save(e, raw, centers, spectra, masses)
    plot_all(centers, spectra, masses, sigma_n)
    print(f"saved: {HERE / 'migdal_Si_raw.csv'}")
    print(f"saved: {HERE / 'migdal_Si_spectra.csv'}")
    print(f"saved: {HERE / 'migdal_Si_overlay.png'}")
    return centers, spectra


def smoke(qedark=None):
    ours = load_ff(FF_DIR / "C.Si137.dat", "C.Si137", 1250, 500, half=0.1)
    qdark = load_ff(qd_path(qedark), "QEdark", half=0.04)
    e = 10.5 * nu.eV
    mass = 0.1 * nu.GeV
    sigma_n = 1.0e-38 * nu.cm**2
    values = {
        "ours": rate_tab(e, sigma_n, mass, ours, nq=48),
        "qedark": rate_tab(e, sigma_n, mass, qdark, nq=48),
    }
    values["lindhard"] = rate_lin(e, sigma_n, mass, loss_lin(e, nk=64), nr=48)
    shown = {name: value * e * nu.kg * nu.year for name, value in values.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in shown.values()):
        raise RuntimeError(f"smoke test failed: {shown}")
    print("smoke test passed")
    for name, value in shown.items():
        print(f"  {name:8s} dR/dlnE(10.5 eV) = {value:.6g} / (kg yr)")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qedark", help="path to QEdark Si_f2.txt")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--de", type=float, default=0.1, help="internal energy step [eV]")
    parser.add_argument("--emax", type=float, default=60.0, help="maximum energy [eV]")
    parser.add_argument("--nq", type=int, default=1000, help="tabulated-rate q points")
    parser.add_argument("--nk", type=int, default=800, help="Lindhard k points")
    parser.add_argument("--nr", type=int, default=400, help="nuclear-recoil points")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    cfg = get_args()
    if cfg.smoke:
        smoke(cfg.qedark)
        return
    run(cfg.qedark, max(1, cfg.jobs), cfg.de, cfg.emax, cfg.nq, cfg.nk, cfg.nr)


if __name__ == "__main__":
    main()
