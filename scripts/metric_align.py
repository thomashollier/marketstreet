"""Metric, levelled, X-aligned copy of a reconstruction (Market Street, sfm_global27/fixed_lens).

    python scripts/metric_align.py [model_dir] [--out DIR]

Frame (right-handed, metres):
  +X along the street from the first camera toward the Ferry Building, = metres along data/market_street_profile.csv
  +Z up (true vertical: pitch fitted so the camera path follows the USGS profile; roll from the cameras' mean down)
  +Y = Z x X (to the left of travel), 0 on the fitted path line
  Z = USGS elevation (offset chosen so the road surface measured from the cloud sits on the profile)
Writes points.ply, poses.npz (camera-to-world, OpenCV camera axes, + `align_matrix` 4x4 similarity
original -> metric, + `scale`), trajectory.txt (TUM), as scripts/align_to_path.py does.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from align_to_path import read_ply, write_ply, quat_xyzw   # noqa: E402

M_PER_UNIT = 139.9          # Call Building 4.37 u = 1425 m, Ferry facade 13.055 u = 2640 m
FIX_U, FIX_M = 4.37, 1425.0  # along-path units from the first camera -> profile metres


def road_profile(P, C, half_width=12.0, below=1.0, step=30.0, min_pts=200):
    """Road height per slice (metric frame): lowest surface (10th pct z) near the track, under the camera."""
    xs, zs = [], []
    order = np.argsort(P[:, 0])
    Ps = P[order]
    for lo in np.arange(C[:, 0].min(), C[:, 0].max() - step / 2, step):
        hi = lo + step
        Q = Ps[np.searchsorted(Ps[:, 0], lo):np.searchsorted(Ps[:, 0], hi)]
        cy = np.interp(Q[:, 0], C[:, 0], C[:, 1])
        cz = np.interp(Q[:, 0], C[:, 0], C[:, 2])
        Q = Q[(np.abs(Q[:, 1] - cy) < half_width) & (Q[:, 2] < cz - below)]
        if len(Q) >= min_pts:
            xs.append((lo + hi) / 2)
            zs.append(np.percentile(Q[:, 2], 10))
    return np.array(xs), np.array(zs)


def chord_dev(x, z):
    return z - (z[0] + (z[-1] - z[0]) * (x - x[0]) / (x[-1] - x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(ROOT / "outputs/SF_1906_4k_gc/sfm_global27/fixed_lens"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=str(ROOT / "data/market_street_profile.csv"))
    a = ap.parse_args()
    src = pathlib.Path(a.model)
    out = pathlib.Path(a.out) if a.out else src / "metric"

    z = np.load(src / "poses.npz")
    poses = z["poses"].astype(np.float64)
    C = poses[:, :3, 3]
    prof = np.genfromtxt(a.profile, delimiter=",", names=True)
    elev = lambda xm: np.interp(xm, prof["distance_m"], prof["elevation_m"])

    # axes before pitch: X0 = path line, Z0 = up from the cameras' mean down (roll), Y0 = Z0 x X0
    d = np.linalg.svd(C - C.mean(0))[2][0]
    if d @ (C[-1] - C[0]) < 0:
        d = -d
    down = poses[:, :3, 1].mean(0)
    down -= (down @ d) * d
    z0 = -down / np.linalg.norm(down)
    x0 = d
    y0 = np.cross(z0, x0)
    R0 = np.stack([x0, y0, z0])
    L0 = (C - C[0]) @ R0.T * M_PER_UNIT          # metric, first camera at origin

    # pitch about Y: camera Z should follow profile elevation + constant height (least squares)
    def frame(theta):
        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # positive theta tips +X up... sign solved by search
        return Ry

    x_off = FIX_M - FIX_U * M_PER_UNIT          # profile metres of the first camera

    def cost(theta):
        Lc = L0 @ frame(theta).T
        r = Lc[:, 2] - elev(Lc[:, 0] + x_off)
        return np.sum((r - r.mean()) ** 2), r.mean()

    th = np.linspace(np.radians(-3), np.radians(3), 601)
    for _ in range(4):
        cs = [cost(t)[0] for t in th]
        b = th[int(np.argmin(cs))]
        w = th[1] - th[0]
        th = np.linspace(b - 2 * w, b + 2 * w, 201)
    theta = b
    R = frame(theta) @ R0
    Cm = (C - C[0]) @ R.T * M_PER_UNIT
    Cm[:, 0] += x_off
    xyz, rgb = read_ply(src / "points.ply")
    Pm = (xyz - C[0]) @ R.T * M_PER_UNIT
    Pm[:, 0] += x_off

    # Y = 0 on the fitted path line (lateral offset of the line in this frame)
    y_line = np.polyval(np.polyfit(Cm[:, 0], Cm[:, 1], 1), Cm[:, 0]).mean()
    Cm[:, 1] -= y_line
    Pm[:, 1] -= y_line
    # Z: road surface from the cloud onto the profile
    rx, rz = road_profile(Pm, Cm)
    z_off = np.median(elev(rx) - rz)
    Cm[:, 2] += z_off
    Pm[:, 2] += z_off
    rz = rz + z_off

    # similarity original -> metric
    A = np.eye(4)
    A[:3, :3] = M_PER_UNIT * R
    A[:3, 3] = -M_PER_UNIT * R @ C[0] + np.array([x_off, -y_line, z_off])
    new = poses.copy()
    new[:, :3, :3] = R @ poses[:, :3, :3]
    new[:, :3, 3] = Cm

    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "points.ply", Pm.astype(np.float32), rgb)
    np.savez(out / "poses.npz", **{**{k: z[k] for k in z.files}, "poses": new.astype(np.float32),
                                   "align_matrix": A, "scale": M_PER_UNIT})
    with open(out / "trajectory.txt", "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (metres; X along Market St profile, Y left, Z up = USGS elevation)\n")
        for i, P in enumerate(new):
            f.write(f"{i:.6f} " + " ".join(f"{v:.6f}" for v in (*P[:3, 3], *quat_xyzw(P[:3, :3]))) + "\n")

    # report
    h = Cm[:, 2] - elev(Cm[:, 0])
    pl = np.linalg.norm(np.diff(Cm, axis=0), axis=1).sum()
    print(f"cameras {len(Cm)}, points {len(Pm):,}, scale {M_PER_UNIT} m/unit")
    print(f"path: x {Cm[0, 0]:.1f} -> {Cm[-1, 0]:.1f} m (extent {Cm[-1, 0] - Cm[0, 0]:.1f} m), cumulative length {pl:.1f} m")
    print(f"pitch applied about Y: {np.degrees(theta):+.4f} deg; roll from cameras' mean down")
    print(f"camera height above profile: median {np.median(h):.3f} m, IQR {np.percentile(h, 25):.3f}-{np.percentile(h, 75):.3f} m")
    print(f"RMS of camera Z - (profile + median height): {np.sqrt(np.mean((h - np.median(h)) ** 2)):.3f} m")
    print(f"camera lateral Y: RMS {np.sqrt(np.mean(Cm[:, 1] ** 2)):.3f} m, max |Y| {np.abs(Cm[:, 1]).max():.2f} m")
    e = elev(rx)
    dr, dp = chord_dev(rx, rz), chord_dev(rx, e)
    i, j = np.argmax(np.abs(dr)), np.argmax(np.abs(dp))
    print(f"road ({len(rx)} slices of 30 m): RMS vs profile {np.sqrt(np.mean((rz - e) ** 2)):.2f} m, max {np.abs(rz - e).max():.2f} m")
    print(f"crest above chord: road {dr[i]:+.2f} m at x={rx[i]:.0f}; real {dp[j]:+.2f} m at x={rx[j]:.0f}")
    print(f"road end-to-end drop: {rz[0] - rz[-1]:.2f} m; real {e[0] - e[-1]:.2f} m")
    print(f"all finite: {bool(np.isfinite(Pm).all() and np.isfinite(new).all())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
