"""Compare the legacy Lin2 table with three Lindhard calculations."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

_MPL = os.environ.get("MPLCONFIGDIR")
if not _MPL or not os.access(_MPL, os.W_OK):
    _MPL = str(Path(tempfile.gettempdir()) / "lin2-check-mpl")
    Path(_MPL).mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _MPL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import migdal_lindhard as ml


HERE = Path(__file__).resolve().parent
LIN2 = HERE.parent / "data/form_factors/Lin2_HgCdTe_lagacy.txt"
QS = (0.01, 0.1, 0.5, 1.0)


def calc(eps):
    """Calculate |epsilon|^2 on the 800 x 300 notebook grid."""
    e = ml.E_grid[None, :]
    q = ml.q_grid[:, None]
    return np.abs(eps(e, q)) ** 2


def pos(index):
    iq, ie = index
    q = ml.q_grid[iq] / (ml.nu.aEM * ml.nu.mElectron)
    e = ml.E_grid[ie] / ml.nu.eV
    return q, e


def report(name, ref, got):
    delta = got - ref
    rel = np.abs(delta) / ref
    ratio = got / ref

    print(f"\n{name}: {got.min():.8g} .. {got.max():.8g}")
    print(f"median relative error: {np.median(rel):.6g}")
    print(f"mean relative error:   {np.mean(rel):.6g}")
    print(f"99% relative error:    {np.quantile(rel, 0.99):.6g}")
    print(f"max relative error:    {np.max(rel):.6g}")
    print(f"log10-ratio RMS:       {np.sqrt(np.mean(np.log10(ratio) ** 2)):.6g} dex")
    print(f"within 10%:            {np.mean(rel < 0.1):.3%}")
    print(f"within factor 2:       {np.mean((ratio > 0.5) & (ratio < 2.0)):.3%}")

    for label, array, fn in (
        ("reference minimum", ref, np.argmin),
        (f"{name} minimum", got, np.argmin),
        ("largest ratio", ratio, np.argmax),
    ):
        index = np.unravel_index(fn(array), array.shape)
        q, e = pos(index)
        print(
            f"{label}: q={q:.3g} alpha*m_e, E={e:.3g} eV, "
            f"reference={ref[index]:.8g}, {name}={got[index]:.8g}"
        )


def panel(ax, e, q, ref, wrong, free8, free32, value):
    iq = int(np.argmin(np.abs(q - value)))
    ax.plot(e, ref[iq], color="black", lw=2.0, label="table")
    ax.plot(e, wrong[iq], "--", color="tab:orange", lw=1.6,
            label="wrong result(1/2 addition)")
    ax.plot(e, free8[iq], "-.", color="tab:blue", lw=1.6,
            label="8 free e reproduce")
    ax.plot(e, free32[iq], ":", color="tab:green", lw=1.8,
            label="32 free e reproduce")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E$ [eV]")
    ax.set_ylabel(r"$|\epsilon|^2$")
    ax.set_title(rf"$q={q[iq]:g}\,\alpha m_e$")
    ax.grid(alpha=0.2)


def draw(ref, wrong, free8, free32, output):
    e = ml.E_grid / ml.nu.eV
    q = ml.q_grid / (ml.nu.aEM * ml.nu.mElectron)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    for ax, value in zip(axes.flat, QS):
        panel(ax, e, q, ref, wrong, free8, free32, value)
    axes[0, 0].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    outputs = [output]
    for value in QS:
        fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        panel(ax, e, q, ref, wrong, free8, free32, value)
        ax.legend()
        tag = f"{value:g}".replace(".", "p")
        path = output.with_name(f"{output.stem}_q{tag}{output.suffix}")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=LIN2)
    parser.add_argument("--out", type=Path, default=HERE / "lin2_check.png")
    parser.add_argument("--save", type=Path, help="optional prefix for calculated tables")
    return parser.parse_args()


def main():
    cfg = args()
    ref = np.loadtxt(cfg.table)
    expected = (ml.N_q, ml.N_E)
    if ref.shape != expected:
        raise ValueError(f"{cfg.table}: shape {ref.shape}, expected {expected}")

    wrong = calc(ml.eps_legacy)
    _, kf8, vf8, wp8 = ml._gas(8)
    free8 = calc(lambda e, q: ml._eps(e, q, None, 0.5, kf8, vf8, wp8))
    free32 = calc(ml.eps_new)
    print(f"grid: q={ref.shape[0]} x E={ref.shape[1]}")
    print("q: 0.01..8 alpha*m_e, dq=0.01 alpha*m_e")
    print("E: 0.05..15 eV, dE=0.05 eV")
    print(f"table: {cfg.table}")
    print(f"range: {ref.min():.8g} .. {ref.max():.8g}")
    report("wrong result(1/2 addition)", ref, wrong)
    report("8 free e reproduce", ref, free8)
    report("32 free e reproduce", ref, free32)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    for output in draw(ref, wrong, free8, free32, cfg.out):
        print(f"saved: {output}")
    if cfg.save:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        wrong_out = cfg.save.with_name(cfg.save.name + "_wrong.txt")
        free8_out = cfg.save.with_name(cfg.save.name + "_free8.txt")
        free32_out = cfg.save.with_name(cfg.save.name + "_free32.txt")
        np.savetxt(wrong_out, wrong)
        np.savetxt(free8_out, free8)
        np.savetxt(free32_out, free32)
        print(f"saved: {wrong_out}")
        print(f"saved: {free8_out}")
        print(f"saved: {free32_out}")


if __name__ == "__main__":
    main()
