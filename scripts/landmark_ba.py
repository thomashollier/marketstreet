"""Add a tracked landmark (e.g. the Ferry Building tower top) to a COLMAP model as one 3D point seen in every frame,
then bundle-adjust, so a single fixed point visible along the whole ride holds the rotation steady.

    python scripts/landmark_ba.py --model out/sfm_main/fixed_lens --landmark data/ferry_tower_top.csv \\
        --video-size 1618 1174 --out out/sfm_main/landmark

The landmark's 3D position is unknown (no height or distance assumed): it is solved with everything else. Its
observations come from track_landmark.py (frame, u, v in --video-size pixels, ok); image names are f<frame>.jpg as
written by sfm_colmap.py. The point is added --weight times (identical copies), so its ~1000 observations aren't
drowned by the model's millions; the lens stays fixed. Writes the adjusted model plus poses.npz / points.ply like
sfm_colmap.py, and prints how far the landmark's reprojections were off before and after.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sfm_colmap import export   # noqa: E402


def landmark_obs(csv, rec, video_size):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    d = d[d["ok"] > 0]
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    cam = next(iter(rec.cameras.values()))
    sx, sy = cam.width / video_size[0], cam.height / video_size[1]
    obs = []
    for f, u, v in zip(d["frame"].astype(int), d["u"], d["v"]):
        im = by_name.get(f"f{f:06d}.jpg")
        if im is not None:
            obs.append((im.image_id, np.array([u * sx, v * sy])))
    return obs


def init_point(rec, obs):
    """Starting position: the least-squares point nearest to all rays, or, if it fits better, the best of a set of
    candidates along several cameras' rays at a range of distances (the rays are nearly parallel, and a bent model
    can put the least-squares point behind the cameras). Scored by median reprojection error; BA refines it."""
    O, D = [], []
    for iid, uv in obs:
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        ray = np.append(cam.cam_from_img(uv), 1.0)
        R = im.cam_from_world().rotation.matrix()
        D.append(R.T @ ray / np.linalg.norm(ray))
        O.append(im.projection_center())
    O, D = np.array(O), np.array(D)
    M = sum(np.eye(3) - np.outer(d, d) for d in D)
    b = sum((np.eye(3) - np.outer(d, d)) @ o for o, d in zip(O, D))
    cands = [np.linalg.solve(M, b)]
    span = np.linalg.norm(O[-1] - O[0])
    for k in np.linspace(0, len(O) - 1, 15).astype(int):
        for t in np.geomspace(0.05, 3.0, 25):
            cands.append(O[k] + D[k] * t * span)
    score = [np.median(reproj(rec, obs, X)) for X in cands]
    return cands[int(np.argmin(score))]


def reproj(rec, obs, X):
    e = []
    for iid, uv in obs:
        im = rec.images[iid]
        p = rec.cameras[im.camera_id].img_from_cam(im.cam_from_world() * X)
        e.append(np.inf if p is None else np.linalg.norm(p - uv))      # None: behind the camera
    return np.array(e)


def add_to_text_model(src_dir, dst_dir, obs, X, copies):
    """Append the landmark's observations to images.txt and `copies` identical 3D points to points3D.txt."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(src_dir))
    dst_dir.mkdir(parents=True, exist_ok=True)
    rec.write_text(str(dst_dir))
    lines = (dst_dir / "images.txt").read_text().splitlines()
    pid0 = max(rec.point3D_ids()) + 1
    by_img = {}
    for iid, uv in obs:
        by_img.setdefault(iid, []).append(uv)
    tracks = {c: [] for c in range(copies)}
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or not ln.strip():
            out.append(ln)
            i += 1
            continue
        head = ln.split()
        iid = int(head[0])
        pts = lines[i + 1]
        n_old = len(pts.split()) // 3
        extra = []
        for uv in by_img.get(iid, []):
            for c in range(copies):
                tracks[c].append((iid, n_old + len(extra)))
                extra.append(f"{uv[0]:.3f} {uv[1]:.3f} {pid0 + c}")
        out += [ln, (pts + " " + " ".join(extra)).strip()]
        i += 2
    (dst_dir / "images.txt").write_text("\n".join(out) + "\n")
    with open(dst_dir / "points3D.txt", "a") as fh:
        for c in range(copies):
            tr = " ".join(f"{iid} {k}" for iid, k in tracks[c])
            fh.write(f"{pid0 + c} {X[0]} {X[1]} {X[2]} 255 0 0 1.0 {tr}\n")
    return pid0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="COLMAP model folder (e.g. sfm_colmap.py's fixed_lens/)")
    ap.add_argument("--landmark", required=True, help="CSV from track_landmark.py")
    ap.add_argument("--video-size", type=int, nargs=2, required=True, metavar=("W", "H"), help="pixel size the CSV refers to")
    ap.add_argument("--weight", type=int, default=20, help="identical copies of the landmark point")
    ap.add_argument("--focal", type=float, default=None, help="set the focal length (pixels at the model's image size) first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pycolmap

    rec = pycolmap.Reconstruction(a.model)
    if a.focal:
        cam = next(iter(rec.cameras.values()))
        p = cam.params.copy()
        p[0] = a.focal
        cam.params = p
    obs = landmark_obs(a.landmark, rec, a.video_size)
    X = init_point(rec, obs)
    e0 = reproj(rec, obs, X)
    print(f"[landmark] {len(obs)} observations in {len(rec.images)} images; before: reprojection median "
          f"{np.median(e0):.1f}px, 90% {np.percentile(e0, 90):.1f}px", flush=True)
    tmp = pathlib.Path(a.out) / "_with_landmark_txt"
    if a.focal:
        (pathlib.Path(a.out) / "_refocused").mkdir(parents=True, exist_ok=True)
        rec.write(str(pathlib.Path(a.out) / "_refocused"))
        src = pathlib.Path(a.out) / "_refocused"
    else:
        src = pathlib.Path(a.model)
    pid0 = add_to_text_model(src, tmp, obs, X, a.weight)
    rec = pycolmap.Reconstruction(str(tmp))
    bo = pycolmap.BundleAdjustmentOptions()
    bo.refine_focal_length = False
    bo.refine_extra_params = False
    bo.refine_principal_point = False
    for _ in range(2):
        pycolmap.bundle_adjustment(rec, bo)
    X1 = rec.points3D[pid0].xyz
    e1 = reproj(rec, obs, X1)
    print(f"[landmark] after: reprojection median {np.median(e1):.2f}px, 90% {np.percentile(e1, 90):.2f}px; "
          f"model mean reprojection {rec.compute_mean_reprojection_error():.3f}px", flush=True)
    export(rec, pathlib.Path(a.out), len(rec.images))


if __name__ == "__main__":
    main()
