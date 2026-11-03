"""How bowed is a reconstruction that should be a level, straight street?

    python scripts/check_level.py outputs/<movie>/LoGeR_star          # or .../lingbot-map, .../fused
    python scripts/check_level.py outputs/<movie>/LoGeR_star --trim-end 0.02   # drop a turn at the end of the line

Needs points.ply + poses.npz in the folder (aligned or not: it aligns in memory the same
way align_to_path.py does, path along +Z, y down). The models' scale is arbitrary, so nothing here is a
fixed distance: results are given as a fraction of the camera path length and in camera heights (the
camera's height above the road, measured from the cloud), and the road search is scaled by that height.

Two measurements, side view (z along the street, y vertical):
  camera path : the height of the camera centres along the street
  ground      : the road surface height along the street. Per slice along the street, relative to the
                camera's own height in that slice (not one height for the whole path, which is wrong once
                the path bends): pass 1 takes the lowest surface below the camera in a strip as wide as a
                slice and gives the camera height (median road - camera); pass 2 repeats with a strip
                --band camera heights either side of the track, keeping points at least --below camera
                heights under the camera.
Each is reported as a "sag": a smooth fit compared with the straight line joining its two ends. A level
street gives ~0 for both (under 0.5% of path length = level). "bowl" = the middle of the street is lower
than its ends, "hill" = higher. Also reported: the same for the plan view (x, left-right) and the change in
ground slope from start to end (a pitch drift shows up as a slope change).
With --profile (a real ground profile, e.g. from usgs_profile.py), the path is scaled to the real distance it
covers (--profile-range) and the road's height above the line joining its ends is compared, in metres, with the
real street's; the camera height this implies is printed as a check.
Writes <folder>/level_check.png if matplotlib is installed (side view in camera heights, or metres with
--profile, exaggerated).
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_to_path import alignment, read_ply   # noqa: E402


def ground_profile(X, C, bins=60, band=None, below=0.0, min_pts=200):
    """Road height per slice along the street, relative to the camera in that slice (y points down).
    band: half-width of the strip around the track (scene units); None = one slice length.
    below: keep only points at least this far under the camera (scene units).
    Returns slice centres z, camera height y and road height y per slice."""
    cz = C[:, 2]
    edges = np.linspace(cz.min(), cz.max(), bins + 1)
    band = edges[1] - edges[0] if band is None else band
    order = np.argsort(X[:, 2])
    Xs = X[order]
    zc, cam, road = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cz >= lo) & (cz < hi)
        if not m.any():
            continue
        P = Xs[np.searchsorted(Xs[:, 2], lo):np.searchsorted(Xs[:, 2], hi)]
        # each point against the camera at its own position along the street, so a slope inside the slice
        # doesn't read as extra height
        dy = P[:, 1] - np.interp(P[:, 2], cz, C[:, 1])
        dx = P[:, 0] - np.interp(P[:, 2], cz, C[:, 0])
        dy = dy[(np.abs(dx) < band) & (dy > below)]
        if len(dy) >= min_pts:
            z = (lo + hi) / 2
            zc.append(z)
            cam.append(np.interp(z, cz, C[:, 1]))
            road.append(cam[-1] + np.percentile(dy, 90))      # the lowest surface near the track
    return np.array(zc), np.array(cam), np.array(road)


def describe(name, z, v, h=None):
    """Print the sag of v(z): a smooth (quadratic) fit compared with the straight line joining its two ends.
    Returns the signed sag in path lengths (y points down: positive = middle lower than the ends = bowl)."""
    zz = np.linspace(z.min(), z.max(), 101)
    fit = np.polyval(np.polyfit(z, v, 2), zz)
    dev = fit - (fit[0] + (fit[-1] - fit[0]) * (zz - zz[0]) / (zz[-1] - zz[0]))
    i = int(np.argmax(np.abs(dev)))
    sag = float(dev[i])
    kind = ("level" if abs(sag) < 0.005 else
            "BOWL: middle lower than the line joining the ends" if sag > 0 else "HILL: middle higher than the line joining the ends")
    ch = f" = {abs(sag) / h:5.1f} camera heights" if h else ""
    print(f"  {name:<12s} sag {100 * abs(sag):5.2f}% of path length{ch}  -> {kind}   [at {100 * (zz[i] - zz[0]) / (zz[-1] - zz[0]):.0f}% along]")
    return sag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="folder with points.ply and poses.npz")
    ap.add_argument("--band", type=float, default=6.0, help="half width of the road strip either side of the track, in camera heights")
    ap.add_argument("--below", type=float, default=0.5, help="road points must be at least this many camera heights under the camera")
    ap.add_argument("--bins", type=int, default=60, help="slices along the street")
    ap.add_argument("--trim-end", type=float, default=0.0, help="drop this fraction of cameras at the end (e.g. a turn at the end of the line)")
    ap.add_argument("--profile", default=None, help="real ground profile CSV (distance_m, elevation_m), e.g. from usgs_profile.py")
    ap.add_argument("--profile-range", type=float, nargs=2, default=None, metavar=("START_M", "END_M"),
                    help="where the first and last camera are along the profile, metres (default: its whole length)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)

    poses = np.load(d / "poses.npz")["poses"].astype(np.float64)
    if a.trim_end > 0:
        poses = poses[: int(len(poses) * (1 - a.trim_end))]
    A, info = alignment(poses)
    C = (A @ poses)[:, :3, 3]
    L = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    xyz, _ = read_ply(d / "points.ply")
    X = (xyz @ A[:3, :3].T + A[:3, 3]) / L
    C = C / L
    cz = C[:, 2]
    keep = (X[:, 2] >= cz.min()) & (X[:, 2] <= cz.max())
    X = X[keep]
    print(f"{len(poses)} cameras, {len(X):,} points along the path, path length {L:.3f} scene units\n")

    # pass 1: camera height above the road, from a strip one slice wide
    z1, c1, r1 = ground_profile(X, C, bins=a.bins)
    h = float(np.median(r1 - c1)) if len(z1) >= 6 else None
    gh = None
    if h:
        zc, cam, road = ground_profile(X, C, bins=a.bins, band=a.band * h, below=a.below * h)
        h = float(np.median(road - cam)) if len(zc) >= 6 else h
        print(f"camera height above the road: {100 * h:.3f}% of path length (path = {1 / h:.0f} camera heights)\n")

    print("camera path, side view (y) and plan view (x):")
    describe("side (y)", cz, C[:, 1], h)
    describe("plan (x)", cz, C[:, 0], h)

    if not h or len(zc) < 6:
        print("\nground: not enough points under/around the path to measure the road "
              "(try --band 12, --below 0.2, or check the cloud has the road)")
    else:
        print("\nroad surface height along the street (side view):")
        gh = describe("road (y)", zc, road, h)
        p = np.polyfit(zc, road, 2)
        ang = lambda zz: np.degrees(np.arctan(-(2 * p[0] * zz + p[1])))
        print(f"  ground slope goes from {ang(zc.min()):+.1f} deg at the start to {ang(zc.max()):+.1f} deg at the end "
              f"(up positive; a level street is ~0 throughout)")

    real = None
    if a.profile and gh is not None:
        # scale the path to the real distance it covers, then compare heights above the line joining the ends
        P = np.genfromtxt(a.profile, delimiter=",", names=True)
        r0, r1 = a.profile_range or (float(P["distance_m"][0]), float(P["distance_m"][-1]))
        Lm = r1 - r0                                             # metres per path length
        dist = lambda z: r0 + (z - cz.min()) / (cz.max() - cz.min()) * Lm
        elev = np.interp(dist(zc), P["distance_m"], P["elevation_m"])
        dep_real = elev - (elev[0] + (elev[-1] - elev[0]) * (zc - zc[0]) / (zc[-1] - zc[0]))
        dep_rec = -(road - (road[0] + (road[-1] - road[0]) * (zc - zc[0]) / (zc[-1] - zc[0]))) * Lm
        real = (dep_real, Lm)
        i, j = np.argmax(np.abs(dep_real)), np.argmax(np.abs(dep_rec))
        print(f"\ncompared with the real ground profile ({a.profile}, {r0:.0f}-{r1:.0f} m, path scaled to {Lm:.0f} m):")
        print(f"  camera height this implies: {h * Lm:.2f} m above the road")
        print(f"  real street:     {dep_real[i]:+6.1f} m from the line joining its ends at most, at {100 * (zc[i] - zc[0]) / (zc[-1] - zc[0]):.0f}% along")
        print(f"  reconstruction:  {dep_rec[j]:+6.1f} m from the line joining its ends at most, at {100 * (zc[j] - zc[0]) / (zc[-1] - zc[0]):.0f}% along")
        print(f"  difference: RMS {np.sqrt(np.mean((dep_rec - dep_real) ** 2)):.1f} m, max {np.abs(dep_rec - dep_real).max():.1f} m "
              f"({100 * np.abs(dep_rec - dep_real).max() / Lm:.2f}% of the distance)")

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(install matplotlib for a picture: pip install matplotlib)")
            return
        u = 1.0 / real[1] if real else (h if h else 1.0)    # plot unit: metres, camera heights or path lengths
        unit = "metres" if real else ("camera heights" if h else "path lengths")
        ref_z, ref_y = (zc, road) if gh is not None else (cz, C[:, 1])
        line = lambda z: ref_y[0] + (ref_y[-1] - ref_y[0]) * (z - ref_z[0]) / (ref_z[-1] - ref_z[0])
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))
        sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 300000), replace=False)]
        near = np.abs(sub[:, 0] - np.interp(sub[:, 2], cz, C[:, 0])) < (a.band * u if h else 0.02)
        ax[0].scatter(sub[near, 2], (sub[near, 1] - line(sub[near, 2])) / u, s=0.05, c="0.8", rasterized=True)
        ax[0].plot(cz, (C[:, 1] - line(cz)) / u, "r-", label="camera path")
        if gh is not None:
            ax[0].plot(zc, (road - line(zc)) / u, "b-", lw=2, label="road surface")
        if real:
            ax[0].plot(zc, -real[0], "g-", lw=2, label="real street (ground profile)")
        ax[0].axhline(0, color="k", ls="--", lw=1, label="straight line joining the road's ends" if gh is not None else "straight line")
        dev = np.r_[(C[:, 1] - line(cz)) / u, -real[0] if real else []]
        lim = max(3.0, 1.3 * np.abs(dev).max())
        ax[0].set_ylim(lim, -lim)                            # y down: flip so up is up
        ax[0].set_ylabel(f"height, {unit}\n(up is up)")
        ax[0].set_title("side view, vertically exaggerated; a level street lies on the dashed line"
                        + ("; the real street is green" if real else ""))
        ax[0].legend(loc="best", fontsize=8)
        ax[1].scatter(sub[:, 2], sub[:, 0] / u, s=0.05, c="0.8", rasterized=True)
        ax[1].plot(cz, C[:, 0] / u, "r-")
        ax[1].set_xlabel("distance along the street (path lengths)")
        xl = max(3.0, 1.3 * np.abs(C[:, 0] / u).max())
        xl = xl if real else max(xl, a.band + 2)
        ax[1].set_ylim(-xl, xl)
        ax[1].set_ylabel(f"left-right, {unit}")
        ax[1].set_title("plan view")
        out = d / "level_check.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
