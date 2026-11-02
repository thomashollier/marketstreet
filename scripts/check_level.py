"""How bowed is a reconstruction that should be a level, straight street?

    python scripts/check_level.py outputs/<movie>/LoGeR_star          # or .../lingbot-map, .../fused

Needs points.ply + poses.npz in the folder (aligned or not: it aligns in memory the same
way align_to_path.py does, path along +Z, y down). Everything is reported as a fraction of the
camera path length, because the models' scale is arbitrary.

Two measurements, side view (z along the street, y vertical):
  camera path : the height of the camera centres along the street
  ground      : the road surface height along the street (the lowest surface in a band around
                the path)
Each is reported as a "sag": a smooth fit compared with the straight line joining its two ends. A level
street gives ~0 for both (under 0.5% = level). "bowl" = the middle of the street is lower than its ends,
"hill" = higher. Also reported: the same for the plan view (x, left-right) and the change in
ground slope from start to end (a pitch drift shows up as a slope change).
Writes <folder>/level_check.png if matplotlib is installed.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_to_path import alignment, read_ply   # noqa: E402


def ground_profile(X, cz, cam_y, bins=30, band=0.3, below=0.03, min_pts=200):
    """Road height per slice of the street: the lowest surface (y points down) near the path."""
    zs = np.linspace(cz.min(), cz.max(), bins + 1)
    sel = (np.abs(X[:, 0]) < band) & (X[:, 1] > cam_y + below) & (X[:, 2] >= zs[0]) & (X[:, 2] <= zs[-1])
    P = X[sel]
    zc, h = [], []
    for lo, hi in zip(zs[:-1], zs[1:]):
        y = P[(P[:, 2] >= lo) & (P[:, 2] < hi), 1]
        if len(y) >= min_pts:
            zc.append((lo + hi) / 2)
            h.append(np.percentile(y, 85))
    return np.array(zc), np.array(h)


def describe(name, z, v):
    """Print the sag of v(z): a smooth (quadratic) fit compared with the straight line joining its two ends.
    Returns the signed sag in path lengths (y points down: positive = middle lower than the ends = bowl)."""
    zz = np.linspace(z.min(), z.max(), 101)
    fit = np.polyval(np.polyfit(z, v, 2), zz)
    dev = fit - (fit[0] + (fit[-1] - fit[0]) * (zz - zz[0]) / (zz[-1] - zz[0]))
    i = int(np.argmax(np.abs(dev)))
    sag = float(dev[i])
    kind = ("level" if abs(sag) < 0.005 else
            "BOWL: middle lower than the line joining the ends" if sag > 0 else "HILL: middle higher than the line joining the ends")
    print(f"  {name:<12s} sag {100 * abs(sag):5.2f}% of path length  -> {kind}   [at {100 * (zz[i] - zz[0]) / (zz[-1] - zz[0]):.0f}% along]")
    return sag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="folder with points.ply and poses.npz")
    ap.add_argument("--band", type=float, default=0.3, help="half width of the strip around the path used for the road, in path lengths")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)

    poses = np.load(d / "poses.npz")["poses"].astype(np.float64)
    A, info = alignment(poses)
    C = (A @ poses)[:, :3, 3]
    L = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    xyz, _ = read_ply(d / "points.ply")
    X = (xyz @ A[:3, :3].T + A[:3, 3]) / L
    C = C / L
    cz = C[:, 2]
    print(f"{len(poses)} cameras, {len(X):,} points, path length {L:.3f} scene units\n")

    print("camera path, side view (y) and plan view (x):")
    cam_y = describe("side (y)", cz, C[:, 1])
    cam_x = describe("plan (x)", cz, C[:, 0])

    zc, h = ground_profile(X, cz, np.median(C[:, 1]), band=a.band)
    if len(zc) < 6:
        print("\nground: not enough points under/around the path to measure the road "
              "(try --band 0.6, or check the cloud has the road)")
        gh = None
    else:
        print("\nroad surface height along the street (side view):")
        gh = describe("road (y)", zc, h)
        p = np.polyfit(zc, h, 2)
        ang = lambda zz: np.degrees(np.arctan(2 * p[0] * zz + p[1]))
        print(f"  ground slope goes from {ang(zc.min()):+.1f} deg at the start to {ang(zc.max()):+.1f} deg at the end "
              f"(a level street is ~constant)")

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(install matplotlib for a picture: pip install matplotlib)")
            return
        fig, ax = plt.subplots(2, 1, figsize=(11, 7))
        sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 60000), replace=False)]
        ax[0].scatter(sub[:, 2], sub[:, 1], s=0.3, c="0.75")
        ax[0].plot(cz, C[:, 1], "r-", label="camera path")
        if gh is not None:
            ax[0].plot(zc, h, "b-", lw=2, label="road height")
            f = np.polyval(np.polyfit(zc, h, 2), [zc.min(), zc.max()])
            ax[0].plot([zc.min(), zc.max()], f, "b--", label="road: line joining the ends")
        f = np.polyval(np.polyfit(cz, C[:, 1], 2), [cz.min(), cz.max()])
        ax[0].plot([cz.min(), cz.max()], f, "r--", label="camera: line joining the ends")
        ax[0].set_ylim(C[:, 1].mean() - 0.5, C[:, 1].mean() + 0.5)
        ax[0].invert_yaxis()
        ax[0].set_title("side view (down is down); level street = flat lines")
        ax[0].legend(loc="best")
        ax[1].scatter(sub[:, 2], sub[:, 0], s=0.3, c="0.75")
        ax[1].plot(cz, C[:, 0], "r-")
        ax[1].set_xlabel("distance along the street (path lengths)")
        ax[1].set_ylim(-0.5, 0.5)
        ax[1].set_title("plan view (left-right)")
        out = d / "level_check.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
