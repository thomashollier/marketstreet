"""Turn a LoGeR predictions .pt (written by demo_viser.py) into poses + a PLY.

Input keys (per demo_viser.py, batch dim already squeezed):
  points[N,H,W,3]  world-space points      conf[N,H,W]  sigmoid confidence
  images[N,H,W,3]  RGB in [0,1]            camera_poses[N,4,4]  world_from_cam

Outputs (under --out):
  poses.npz   { poses[N,4,4]=world_from_cam, frame_idx[N]=source-video frame index }
  points.ply  binary little-endian, confidence-filtered, randomly subsampled
"""
import argparse
import pathlib

import numpy as np
import torch


def write_ply(path, xyz, rgb):
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf-percentile", type=float, default=20.0)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(a.pred, map_location="cpu", weights_only=False)
    pts, conf = d["points"].numpy(), d["conf"].numpy()
    imgs, poses = d["images"].numpy(), d["camera_poses"].numpy()
    if conf.ndim == pts.ndim:   # saved as [N,H,W,1]
        conf = conf[..., 0]
    n = len(poses)
    print(f"[export] shapes: points{pts.shape} conf{conf.shape} "
          f"images{imgs.shape} poses{poses.shape}", flush=True)

    np.savez(out / "poses.npz",
             poses=poses.astype(np.float32),
             frame_idx=a.start_frame + a.stride * np.arange(n))

    keep = conf >= np.percentile(conf, a.conf_percentile)
    xyz = pts[keep].astype(np.float32)
    rgb = (imgs[keep].clip(0, 1) * 255).astype(np.uint8)
    if len(xyz) > a.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    write_ply(out / "points.ply", xyz, rgb)
    print(f"[export] {n} frames, {len(xyz)} points -> {out}", flush=True)


if __name__ == "__main__":
    main()
