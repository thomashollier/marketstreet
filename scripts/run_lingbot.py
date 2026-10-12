"""Headless LingBot-Map: video -> poses + intrinsics + coloured point cloud.

upstream demo.py always opens a viser viewer, so this reuses its load_images /
load_model / postprocess and replaces the viewer with a file export.

Outputs (under --out):
  poses.npz       poses[N,4,4]=world_from_cam, intrinsics[N,3,3] (for the
                  processed HxW image), image_size=[H,W], frame_idx[N]
  trajectory.txt  TUM format `ts tx ty tz qx qy qz qw`
  points.ply      binary little-endian, confidence-filtered, subsampled
"""
import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.environ.get("LINGBOT_DIR", "/opt/lingbot-map"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from export_loger import write_ply   # noqa: E402  (same PLY writer as the LoGeR path)


def extract_frames(video, out_dir, fps):
    vf = ["-vf", f"fps={fps:g}"] if fps > 0 else []
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, *vf, "-q:v", "2",
                    f"{out_dir}/%06d.jpg"], check=True)


def mat_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()   # x y z w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0, help="sample rate; 0 = every frame")
    ap.add_argument("--mode", default="streaming", choices=["streaming", "windowed"])
    ap.add_argument("--keyframe-interval", type=int, default=0, help="0 = auto")
    ap.add_argument("--num-scale-frames", type=int, default=8)
    ap.add_argument("--camera-num-iterations", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--overlap-size", type=int, default=16)
    ap.add_argument("--conf-percentile", type=float, default=20.0)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    a = ap.parse_args()

    import torch
    import demo   # upstream /opt/lingbot-map/demo.py (main() is guarded)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = tempfile.mkdtemp(prefix="lingbot_frames_")
    extract_frames(a.video, frames, a.fps)
    images, paths, _ = demo.load_images(image_folder=frames, image_ext=".jpg")

    # upstream load_model() reads these attributes off the argparse namespace
    args = argparse.Namespace(
        mode=a.mode, image_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=1024, kv_cache_sliding_window=64,
        num_scale_frames=a.num_scale_frames, use_sdpa=True,
        camera_num_iterations=a.camera_num_iterations, model_path=a.model_path)
    model = demo.load_model(args, device)

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model.aggregator = model.aggregator.to(dtype=dtype)   # as upstream main()
    else:
        dtype = torch.float32
    images = images.to(device)
    n = images.shape[0]

    kf = a.keyframe_interval
    if kf <= 0:   # upstream auto rule: bound the KV cache to ~320 keyframes
        kf = (n + 319) // 320 if (a.mode == "streaming" and n > 320) else 1
    print(f"[lingbot] {n} frames, mode={a.mode}, keyframe_interval={kf}", flush=True)

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if a.mode == "streaming":
            pred = model.inference_streaming(
                images, num_scale_frames=a.num_scale_frames,
                keyframe_interval=kf, output_device=torch.device("cpu"))
        else:
            pred = model.inference_windowed(
                images, window_size=a.window_size, overlap_size=a.overlap_size,
                num_scale_frames=a.num_scale_frames, keyframe_interval=kf,
                output_device=torch.device("cpu"))
    dt = time.time() - t0
    print(f"[lingbot] inference {dt:.1f}s ({n / dt:.2f} FPS)", flush=True)

    imgs_cpu = pred["images"]
    del images
    pred, imgs_cpu = demo.postprocess(pred, imgs_cpu)
    vis = demo.prepare_for_visualization(pred, imgs_cpu)

    # The model emits depth (point head is off by default). Unproject exactly as
    # upstream's viewer does: its `extrinsic` is treated as cam_from_world by
    # unproject_depth_map_to_point_map and by the frustum code (inverted there),
    # whatever demo.postprocess' comment says.
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
    depth, conf = vis["depth"], vis["depth_conf"]
    if conf.ndim == 4:
        conf = conf[..., 0]
    ext, intr = vis["extrinsic"], vis["intrinsic"]
    pts = unproject_depth_map_to_point_map(depth, ext, intr)          # [N,H,W,3]
    imgs = vis["images"]
    if imgs.shape[1] == 3:      # [N,3,H,W] -> [N,H,W,3]
        imgs = imgs.transpose(0, 2, 3, 1)
    print(f"[lingbot] shapes: depth{depth.shape} conf{conf.shape} images{imgs.shape} "
          f"extrinsic{ext.shape} intrinsic{intr.shape}", flush=True)

    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    w2c[:, :3, :4] = ext
    poses = np.linalg.inv(w2c).astype(np.float32)                     # world_from_cam
    np.savez(out / "poses.npz", poses=poses, intrinsics=intr.astype(np.float32),
             image_size=np.array(imgs.shape[1:3]), frame_idx=np.arange(n))
    with open(out / "trajectory.txt", "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, (P, q) in enumerate(zip(poses, mat_to_quat_xyzw(poses[:, :3, :3]))):
            f.write(f"{i:.6f} " + " ".join(f"{v:.6f}" for v in (*P[:3, 3], *q)) + "\n")

    valid = np.isfinite(pts).all(-1) & np.isfinite(conf)
    keep = valid & (conf >= np.percentile(conf[valid], a.conf_percentile))
    xyz = pts[keep].astype(np.float32)
    rgb = (imgs[keep].clip(0, 1) * 255).astype(np.uint8)
    if len(xyz) > a.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    write_ply(out / "points.ply", xyz, rgb)
    print(f"[lingbot] {n} frames, {len(xyz)} points -> {out}", flush=True)


if __name__ == "__main__":
    main()
