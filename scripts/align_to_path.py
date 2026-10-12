"""Rotate a reconstruction so the (straight) camera path runs along +Z.

    python scripts/align_to_path.py outputs/<movie>/lingbot-map

Reads points.ply + poses.npz (from either model), fits a line through the camera
centres, and applies ONE rigid transform to the points and the cameras:
  - the line becomes the Z axis, oriented along the cameras' average forward
    direction, with the first camera at the origin
  - the scene is rolled so the cameras' average "down" is +Y (level horizon)
Cameras are OpenCV convention (x right, y down, z forward), so after alignment
an average camera has an identity rotation: you look straight down +Z.

It does not straighten the path: wobble around the line is kept (and reported).
Writes <dir>/aligned/{points.ply,poses.npz,trajectory.txt}; poses.npz keeps all
original arrays and gains `align_matrix` (4x4, original world -> aligned world).
"""
import argparse
import pathlib

import numpy as np

PLY_DTYPE = np.dtype([("p", "<f4", 3), ("c", "u1", 3)])   # as written by export_loger.write_ply


def read_ply(path):
    raw = pathlib.Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().split("\n")
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    if "binary_little_endian" not in raw[:end].decode() or (len(raw) - end) != n * PLY_DTYPE.itemsize:
        raise SystemExit(f"{path}: not a PLY written by this repo (float xyz + uchar rgb, little-endian)")
    rec = np.frombuffer(raw, PLY_DTYPE, count=n, offset=end)
    return rec["p"].astype(np.float64), rec["c"].copy()


def write_ply(path, xyz, rgb):
    rec = np.empty(len(xyz), PLY_DTYPE)
    rec["p"], rec["c"] = xyz, rgb
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    with open(path, "wb") as f:
        f.write(header + rec.tobytes())


def quat_xyzw(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = np.empty(4)
    q[i] = s / 4
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def alignment(poses):
    """4x4 rigid transform A (original world -> aligned world) and diagnostics."""
    C = poses[:, :3, 3]
    mean = C.mean(0)
    d = np.linalg.svd(C - mean)[2][0]                     # best-fit line direction
    fwd = poses[:, :3, 2].mean(0)                         # mean camera forward (OpenCV +z)
    if d @ fwd < 0:
        d = -d
    down = poses[:, :3, 1].mean(0)
    down = down - (down @ d) * d                          # remove the component along the path
    if np.linalg.norm(down) < 1e-6:
        raise SystemExit("cameras' average 'down' is parallel to the path; cannot define a level")
    y = down / np.linalg.norm(down)
    x = np.cross(y, d)
    R = np.stack([x, y, d])                               # rows: new axes in old coords
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = -R @ C[0]                                  # first camera -> origin
    lateral = (C - mean) - np.outer((C - mean) @ d, d)
    info = {
        "n": len(C),
        "path_length": float(np.linalg.norm(C[-1] - C[0])),
        "rms_wobble": float(np.sqrt((lateral ** 2).sum(1).mean())),
        "fwd_vs_path_deg": float(np.degrees(np.arccos(np.clip(fwd @ d / np.linalg.norm(fwd), -1, 1)))),
    }
    return A, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="folder with points.ply and poses.npz")
    ap.add_argument("--out", help="default: <dir>/aligned")
    a = ap.parse_args()

    src = pathlib.Path(a.dir)
    out = pathlib.Path(a.out) if a.out else src / "aligned"
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(src / "poses.npz")
    poses = z["poses"].astype(np.float64)
    A, info = alignment(poses)
    new_poses = A @ poses

    xyz, rgb = read_ply(src / "points.ply")
    write_ply(out / "points.ply", (xyz @ A[:3, :3].T + A[:3, 3]).astype(np.float32), rgb)

    np.savez(out / "poses.npz", **{**{k: z[k] for k in z.files}, "poses": new_poses.astype(np.float32),
                                   "align_matrix": A})
    with open(out / "trajectory.txt", "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, P in enumerate(new_poses):
            f.write(f"{i:.6f} " + " ".join(f"{v:.6f}" for v in (*P[:3, 3], *quat_xyzw(P[:3, :3]))) + "\n")

    print(f"{info['n']} cameras, path length {info['path_length']:.3f} (scene units)")
    print(f"wobble around the line: RMS {info['rms_wobble']:.4f} "
          f"({100 * info['rms_wobble'] / info['path_length']:.2f}% of path length)")
    print(f"average camera forward vs path direction: {info['fwd_vs_path_deg']:.1f} deg"
          + ("  <-- cameras are not looking along the path" if info["fwd_vs_path_deg"] > 20 else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
