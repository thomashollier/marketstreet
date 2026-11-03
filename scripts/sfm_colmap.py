"""Structure from motion with one shared camera over a stretch of a movie (COLMAP via pycolmap), moving objects masked.

    python scripts/sfm_colmap.py --video reg.mp4 --start 4412 --end 6028 --stride 2 --work out/sfm_pilot

Why: a windowed model (LoGeR) guesses the camera afresh in every window, so small pitch errors add up along a long
ride. Here every frame shares one camera (focal length + radial distortion, solved), features are tracked across
many frames (the sequential matcher also pairs each frame with frames 1, 2, 4, 8, ... ahead, so distant buildings
give long tracks), and bundle adjustment makes all of it agree. Nothing tells it the street is flat.

Steps (each skipped if its output exists, so a run can be resumed):
  1. frames   ffmpeg -> <work>/images/f<frame>.jpg  (frame = index in --video)
  2. masks    torchvision Mask R-CNN: people, vehicles, animals (dilated) -> <work>/masks/<image>.png (0 = ignore)
  3. sfm      SIFT (masked) -> sequential matching -> incremental or global mapping
  4. export   largest model, lens held at --focal-guess -> <work>/fixed_lens/; then one more bundle adjustment
              with focal length + distortion free -> <work>/free_lens/. Each has poses.npz (world_from_cam,
              frame_idx, intrinsics), points.ply, camera.json, sfm_stats.json and the COLMAP model.
              Forward motion down a street can't pin the lens on its own (focal, distortion and depth trade
              off), so compare the two before trusting free_lens.
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np

# COCO classes that move in a 1906 street: person, bicycle, car, motorcycle, bus, train (cable car), truck,
# bird, cat, dog, horse
MOVING = {1, 2, 3, 4, 6, 7, 8, 16, 17, 18, 19}


def extract_frames(video, out, start, end, stride, max_size):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.jpg")):
        return sorted(out.glob("*.jpg"))
    sel = f"select='between(n\\,{start}\\,{end})*not(mod(n-{start}\\,{stride}))'"
    scale = f"scale='if(gt(iw,{max_size}),{max_size},iw)':-2"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"{sel},{scale}", "-fps_mode", "passthrough",
                    "-q:v", "2", "-start_number", "0", str(out / "tmp_%06d.jpg")], check=True)
    files = sorted(out.glob("tmp_*.jpg"))
    for k, f in enumerate(files):
        f.rename(out / f"f{start + k * stride:06d}.jpg")
    return sorted(out.glob("*.jpg"))


def make_masks(images, out, score=0.4, dilate=15):
    """White = use, black = moving object (ignored by feature extraction)."""
    import cv2
    import torch
    import torchvision
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (out / f"{p.name}.png").exists()]
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w).eval().to(dev)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    cover = []
    with torch.no_grad():
        for p in todo:
            img = cv2.imread(str(p))[:, :, ::-1].copy()
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(dev)
            r = model([t])[0]
            keep = (r["scores"] > score) & torch.tensor([int(l) in MOVING for l in r["labels"]], device=dev)
            m = (r["masks"][keep, 0] > 0.5).any(0).cpu().numpy() if keep.any() else np.zeros(img.shape[:2], bool)
            m = cv2.dilate(m.astype(np.uint8), k) > 0
            cover.append(m.mean())
            cv2.imwrite(str(out / f"{p.name}.png"), np.where(m, 0, 255).astype(np.uint8))
    print(f"[sfm] masks: {len(todo)} images, moving objects cover {100 * np.mean(cover):.1f}% on average "
          f"(max {100 * np.max(cover):.1f}%)", flush=True)


def run_sfm(work, images, camera_model, focal_guess, mapper, overlap, max_features, threads):
    import pycolmap
    db = work / "database.db"
    if not db.exists():
        h, w = __import__("cv2").imread(str(images[0])).shape[:2]
        f = focal_guess * w if focal_guess else 1.2 * w
        params = {"SIMPLE_RADIAL": f"{f},{w / 2},{h / 2},0", "RADIAL": f"{f},{w / 2},{h / 2},0,0",
                  "OPENCV": f"{f},{f},{w / 2},{h / 2},0,0,0,0"}[camera_model]
        ro = pycolmap.ImageReaderOptions(camera_model=camera_model, camera_params=params,
                                         mask_path=str(work / "masks"))
        eo = pycolmap.FeatureExtractionOptions()
        eo.sift.max_num_features = max_features
        eo.num_threads = threads
        t = time.time()
        pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo)
        print(f"[sfm] features: {time.time() - t:.0f}s", flush=True)
        po = pycolmap.SequentialPairingOptions(overlap=overlap, quadratic_overlap=True)
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = threads
        t = time.time()
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po)
        print(f"[sfm] matching: {time.time() - t:.0f}s", flush=True)
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    t = time.time()
    if mapper == "global":
        go = pycolmap.GlobalPipelineOptions()
        go.num_threads = threads
        recs = pycolmap.global_mapping(db, work / "images", sparse, options=go)
    else:
        io = pycolmap.IncrementalPipelineOptions()
        io.num_threads = threads
        # lens held at the initial guess while the path is built (two or three frames of forward motion can't pin
        # it, and a runaway distortion term wrecks everything after); refined at the end over all frames
        io.ba_refine_focal_length = False
        io.ba_refine_extra_params = False
        io.ba_refine_principal_point = False      # gate-registered: fixed, and forward motion can't pin it anyway
        io.ignore_watermarks = True               # pairs whose matches don't move are marks on the film, not the street
        # looking straight down a street while moving forward: small triangulation angles, mostly forward motion.
        # COLMAP's defaults (16 deg, forward share <= 0.95) never find a starting pair in such a shot.
        io.mapper.init_min_tri_angle = 4.0
        io.mapper.init_max_forward_motion = 0.995
        recs = pycolmap.incremental_mapping(db, work / "images", sparse, options=io)
    print(f"[sfm] mapping ({mapper}): {time.time() - t:.0f}s, {len(recs)} model(s): "
          f"{[r.num_reg_images() for r in recs.values()]} images", flush=True)
    return max(recs.values(), key=lambda r: r.num_reg_images())


def free_lens_ba(rec):
    """Bundle adjustment over everything with focal length + distortion free (principal point stays)."""
    import pycolmap
    cam0 = next(iter(rec.cameras.values())).params.copy()
    bo = pycolmap.BundleAdjustmentOptions()
    bo.refine_focal_length = True
    bo.refine_extra_params = True
    bo.refine_principal_point = False
    pycolmap.bundle_adjustment(rec, bo)
    cam1 = next(iter(rec.cameras.values())).params
    print(f"[sfm] BA with the lens free: params {np.round(cam0, 4).tolist()} -> {np.round(cam1, 4).tolist()}", flush=True)


def write_ply(path, xyz, rgb):
    """Same binary layout as export_loger.write_ply (which imports torch: kept out of this process)."""
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def export(rec, work, n_images):
    names, poses = [], []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        T = np.eye(4)
        T[:3, :4] = img.cam_from_world().matrix()
        names.append(img.name)
        poses.append(np.linalg.inv(T))
    order = np.argsort(names)
    names = [names[i] for i in order]
    poses = np.array(poses)[order]
    frame_idx = np.array([int(n[1:7]) for n in names])
    cam = next(iter(rec.cameras.values()))
    K = cam.calibration_matrix()
    work.mkdir(parents=True, exist_ok=True)
    rec.write(str(work))                                   # COLMAP binary model
    np.savez(work / "poses.npz", poses=poses.astype(np.float32), frame_idx=frame_idx,
             intrinsics=np.repeat(K[None], len(poses), 0).astype(np.float32),
             image_size=np.array([cam.width, cam.height]))
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()]).astype(np.uint8)
    write_ply(work / "points.ply", xyz.astype(np.float32), rgb)
    track = np.array([p.track.length() for p in rec.points3D.values()])
    fov = np.degrees(2 * np.arctan(cam.width / 2 / cam.params[0]))
    camj = {"model": cam.model.name, "params": [float(x) for x in cam.params], "width": cam.width,
            "height": cam.height, "horizontal_fov_deg": float(fov)}
    stats = {"images_in": n_images, "images_registered": len(poses), "points": len(xyz),
             "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
             "track_length_mean": float(track.mean()), "track_length_p90": float(np.percentile(track, 90)),
             "camera": camj}
    (work / "camera.json").write_text(json.dumps(camj, indent=1))
    (work / "sfm_stats.json").write_text(json.dumps(stats, indent=1))
    print("[sfm] " + json.dumps(stats), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--start", type=int, required=True, help="first frame (index in --video)")
    ap.add_argument("--end", type=int, required=True, help="last frame (inclusive)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=1600, help="longest image side for SfM")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL", choices=["SIMPLE_RADIAL", "RADIAL", "OPENCV"])
    ap.add_argument("--focal-guess", type=float, default=1.43, help="initial focal length / image width (LoGeR implies ~1.43)")
    ap.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    ap.add_argument("--overlap", type=int, default=15, help="sequential neighbours matched (plus 2^k quadratic links)")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--fixed-lens", action="store_true", help="skip the free_lens bundle adjustment")
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    images = extract_frames(a.video, work / "images", a.start, a.end, a.stride, a.max_size)
    print(f"[sfm] {len(images)} frames {a.start}..{a.end} step {a.stride}", flush=True)
    if a.no_masks:
        (work / "masks").mkdir(exist_ok=True)
        for p in images:
            mp = work / "masks" / f"{p.name}.png"
            if not mp.exists():
                import cv2
                h, w = cv2.imread(str(p)).shape[:2]
                cv2.imwrite(str(mp), np.full((h, w), 255, np.uint8))
    else:
        # own process: torch and pycolmap each bring an OpenMP runtime, and two in one process can crash
        import multiprocessing
        proc = multiprocessing.get_context("spawn").Process(target=make_masks, args=(images, work / "masks"))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"mask step failed (exit {proc.exitcode})")
    rec = run_sfm(work, images, a.camera_model, a.focal_guess, a.mapper, a.overlap, a.max_features, a.threads)
    export(rec, work / "fixed_lens", len(images))
    if not a.fixed_lens and rec.num_reg_images() >= 20:
        free_lens_ba(rec)
        export(rec, work / "free_lens", len(images))


if __name__ == "__main__":
    main()
