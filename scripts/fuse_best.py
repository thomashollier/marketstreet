"""Fuse a LoGeR reconstruction keeping the best observation of each surface.

    python scripts/fuse_best.py --pred <raw predictions .pt> --out <dir>

Every pixel of every frame is a candidate surface point. Observations are processed from
nearest to farthest: a point's uncertainty radius is rel_err * depth, and an observation is
dropped if a better (nearer) one already sits within that radius, so a distant, less accurate
view of a surface is replaced by the later close-up but is kept where nothing better exists.
Within one scale, each candidate gets a quality weight

    w = confidence * depth^-p * cos(angle between surface normal and view ray)^2

(depth error grows with distance, and grazing views are poor), after hard filters
(low confidence, depth-discontinuity "flying pixels", the farthest few percent of
each frame, near-grazing). Candidates are then binned into voxels; per voxel only the
best tier survives (weight >= tier * the voxel's best) and those survivors are
averaged. So a facade patch seen from afar early in the shot is replaced by the view
from when the camera was nearest and facing it, but still kept if that is the only view.
Voxels seen by fewer than --min-frames distinct frames are dropped: static surfaces are
seen repeatedly, floaters and moving objects (streetcars, people) land somewhere new in
every frame.

Camera poses are not touched (they stay consistent with the images). Output is in the
input's world frame:  points.ply, poses.npz, fusion_stats.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _tilt_allowance(rn, r, cos, sl_a, sl_b):
    """Relative range change between two neighbouring pixels that a FLAT surface at this tilt would produce
    (about pixel angle * cot(grazing angle)); anything beyond it is a real depth discontinuity."""
    cross = np.linalg.norm(np.cross(rn[sl_a], rn[sl_b]), axis=-1)
    step = np.arctan2(cross, (rn[sl_a] * rn[sl_b]).sum(-1))       # angle between the two pixel rays
    c = np.minimum(cos[sl_a], cos[sl_b])
    return np.abs(r[sl_a] - r[sl_b]) / np.minimum(r[sl_a], r[sl_b]), step * np.sqrt(np.maximum(1 - c ** 2, 0)) / np.maximum(c, 0.05)


def frame_candidates(P, conf, rgb, C, fwd, a, stats):
    """Quality-filtered, weighted candidate points of one frame. P [H,W,3], conf [H,W], rgb [H,W,3] 0..1.
    `stats` accumulates how many pixels each filter removed."""
    ray = P - C
    r = np.linalg.norm(ray, axis=-1)
    z = (ray * fwd).sum(-1)                                      # depth along the camera's forward axis
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    cos = np.abs((n * ray).sum(-1)) / (np.linalg.norm(n, axis=-1) * r + 1e-12)

    valid = np.isfinite(P).all(-1) & np.isfinite(conf) & (z > 0)
    stats["pixels"] += valid.size
    stats["invalid"] += int((~valid).sum())
    rn = ray / np.maximum(r[..., None], 1e-12)
    rs = np.where(valid, r, np.nan)
    edge = np.zeros_like(valid)                                  # flying pixels at depth discontinuities
    with np.errstate(invalid="ignore"):
        for sl_a, sl_b in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                           ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
            rel, allow = _tilt_allowance(rn, rs, cos, sl_a, sl_b)
            hit = rel > a["rtol"] + 2.0 * allow
            edge[sl_a] |= hit
            edge[sl_b] |= hit
    ok = valid & ~edge
    stats["flying_pixels"] += int(valid.sum() - ok.sum())
    before = int(ok.sum())
    ok &= cos >= a["min_cos"]
    stats["grazing"] += before - int(ok.sum())
    if not ok.any():
        return None
    before = int(ok.sum())
    ok &= conf >= np.percentile(conf[ok], a["conf_pct"])
    stats["low_confidence"] += before - int(ok.sum())
    before = int(ok.sum())
    ok &= z <= np.percentile(z[ok], a["far_pct"])
    stats["far"] += before - int(ok.sum())

    s = a["pixel_stride"]
    sub = np.zeros_like(ok)
    sub[::s, ::s] = True
    before = int(ok.sum())
    ok &= sub
    stats["stride_decimation"] += before - int(ok.sum())
    w = np.clip(conf, 0.05, 1.0) * np.maximum(z, 1e-9) ** (-a["p"]) * np.maximum(cos, 1e-3) ** 2
    return P[ok], np.clip(rgb[ok] * 255, 0, 255).astype(np.uint8), w[ok], z[ok], int(ok.sum())


def fuse(xyz, rgb, w, frame, voxel, tier=0.5, min_frames=2, stats=None):
    """Voxel-wise best-tier fusion. Returns (xyz, rgb 0..255) of the surviving, averaged points."""
    ijk = np.floor(xyz / voxel).astype(np.int64)
    ijk -= ijk.min(0)
    dims = ijk.max(0) + 2
    if float(np.prod(dims.astype(float))) > 9e18:
        raise SystemExit(f"voxel {voxel:g} is too small for this scene extent; raise --voxel")
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

    # support: distinct frames per 2x-coarser cell (a fine voxel inherits its cell's support)
    cijk = ijk // 2
    ckey = (cijk[:, 0] * dims[1] + cijk[:, 1]) * dims[2] + cijk[:, 2]
    o = np.lexsort((frame, ckey))
    ck, fr = ckey[o], frame[o]
    first = np.r_[True, (ck[1:] != ck[:-1]) | (fr[1:] != fr[:-1])]
    ucells, cstart = np.unique(ck, return_index=True)
    support = np.add.reduceat(first.astype(np.int64), cstart)[np.searchsorted(ucells, ckey)]

    o = np.argsort(key, kind="stable")
    ks, ws = key[o], w[o].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    vid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ks)]))
    wmax = np.maximum.reduceat(ws, starts)
    supported = support[o] >= min_frames
    best = ws >= tier * wmax[vid]
    if stats is not None:
        stats["seen_by_too_few_frames"] = stats.get("seen_by_too_few_frames", 0) + int((~supported).sum())
        stats["weaker_than_best_tier"] = stats.get("weaker_than_best_tier", 0) + int((supported & ~best).sum())
    keep = best & supported
    wk = ws * keep
    sw = np.add.reduceat(wk, starts)
    good = sw > 0
    out_xyz = np.add.reduceat(wk[:, None] * xyz[o].astype(np.float64), starts, axis=0)[good] / sw[good, None]
    out_rgb = np.add.reduceat(wk[:, None] * rgb[o].astype(np.float64), starts, axis=0)[good] / sw[good, None]
    return out_xyz.astype(np.float32), out_rgb.astype(np.float32)


def fuse_nearest_first(xyz, rgb, w, frame, z, voxel, rel_err, tier=0.5, min_frames=2, stats=None):
    """Nearest-first fusion in depth octaves. Each octave is fused at the fine `voxel` size, but first any
    observation lying within rel_err * its depth of an already accepted (nearer, better) point is dropped."""
    from scipy.spatial import cKDTree
    octave = np.floor(np.log2(np.maximum(z, 1e-9))).astype(int)
    acc_x, acc_c = [], []
    for k in np.unique(octave):
        sel = np.flatnonzero(octave == k)
        if acc_x:
            d, _ = cKDTree(np.concatenate(acc_x)).query(xyz[sel], workers=-1)
            keep = d >= rel_err * z[sel]
            if stats is not None:
                stats["replaced_by_nearer_view"] = stats.get("replaced_by_nearer_view", 0) + int((~keep).sum())
            sel = sel[keep]
        if len(sel):
            fx, fc = fuse(xyz[sel], rgb[sel], w[sel], frame[sel], voxel, tier, min_frames, stats)
            acc_x.append(fx)
            acc_c.append(fc)
    return np.concatenate(acc_x), np.concatenate(acc_c)


def report(st, n_candidates, n_fused, n_written, cap):
    """Human-readable account of what each stage removed (pixel / observation counts)."""
    px = st["pixels"]
    lines = [f"[fuse] {px:,} input pixels"]
    for key, label in (("invalid", "invalid / behind camera"), ("flying_pixels", "flying pixels (depth discontinuity)"),
                       ("grazing", "near-grazing views"), ("low_confidence", "lowest-confidence per frame"),
                       ("far", "farthest per frame"), ("stride_decimation", "skipped by --pixel-stride (not a filter)")):
        lines.append(f"[fuse]   - {st[key]:>12,} ({100 * st[key] / px:5.1f}%) {label}")
    lines.append(f"[fuse] = {n_candidates:,} candidate observations")
    for key, label in (("replaced_by_nearer_view", "replaced by a nearer, better view (--rel-err)"),
                       ("seen_by_too_few_frames", "surfaces seen by too few frames (--min-frames; moving objects)"),
                       ("weaker_than_best_tier", "merged away: weaker than the best view of the same voxel (--tier)")):
        v = st.get(key, 0)
        lines.append(f"[fuse]   - {v:>12,} ({100 * v / max(n_candidates, 1):5.1f}%) {label}")
    lines.append(f"[fuse] = {n_fused:,} fused surface points" + (f"  -> CAPPED to {n_written:,} by --max-points {cap:,} "
                 "(random thinning: raise --max-points)" if n_written < n_fused else ""))
    return "\n".join(lines)


def main():
    import torch
    from export_loger import write_ply

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="LoGeR raw predictions .pt (points, conf, images, camera_poses)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size in scene units; 0 = 0.002 x camera path length")
    ap.add_argument("--tier", type=float, default=0.5, help="keep observations with weight >= tier x the voxel's best")
    ap.add_argument("--min-frames", type=int, default=2, help="drop voxels seen by fewer distinct frames")
    ap.add_argument("--rel-err", type=float, default=0.03,
                    help="depth error as a fraction of depth; sets how far a better nearer point suppresses a "
                         "farther one (0 = single-scale voxel fusion, no nearest-first rule)")
    ap.add_argument("--p", type=float, default=2.0, help="depth exponent: weight ~ depth^-p")
    ap.add_argument("--conf-pct", type=float, default=20.0, help="drop each frame's lowest N%% confidence")
    ap.add_argument("--far-pct", type=float, default=98.0, help="drop each frame's farthest (100-N)%%")
    ap.add_argument("--rtol", type=float, default=0.03,
                    help="relative depth jump between neighbouring pixels tolerated ON TOP of what a flat surface at "
                         "that tilt gives (bigger = fewer pixels dropped as flying pixels)")
    ap.add_argument("--min-cos", type=float, default=0.01,
                    help="only drop views closer to edge-on than this cos(angle) (about 0.6 deg); grazing views are otherwise "
                         "kept and just weighted down")
    ap.add_argument("--pixel-stride", type=int, default=1, help="use every Nth pixel in each direction")
    ap.add_argument("--max-points", type=int, default=4_000_000)
    a = ap.parse_args()
    args = vars(a)

    d = torch.load(a.pred, map_location="cpu", weights_only=False)
    P, conf, img, poses = (d[k].numpy() for k in ("points", "conf", "images", "camera_poses"))
    if conf.ndim == P.ndim:
        conf = conf[..., 0]
    poses = poses.astype(np.float64)
    n = len(poses)
    C = poses[:, :3, 3]
    voxel = a.voxel or 0.002 * float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    print(f"[fuse] {n} frames, voxel {voxel:.5f}", flush=True)

    st = dict.fromkeys(("pixels", "invalid", "flying_pixels", "grazing", "low_confidence", "far", "stride_decimation"), 0)
    X, R, W, Z, F, kept = [], [], [], [], [], []
    for i in range(n):
        res = frame_candidates(P[i], conf[i], img[i], C[i], poses[i, :3, 2], args, st)
        if res is None:
            kept.append(0)
            continue
        x, c, w, zz, k = res
        X.append(x), R.append(c), W.append(w), Z.append(zz), F.append(np.full(len(x), i, np.int32)), kept.append(k)
    xyz, rgb, w, z, frame = (np.concatenate(v) for v in (X, R, W, Z, F))
    del P, conf, img, X, R, W, Z, F
    print(f"[fuse] {len(xyz):,} candidate observations after per-frame filters", flush=True)

    if a.rel_err > 0:
        fx, fc = fuse_nearest_first(xyz, rgb, w, frame, z, voxel, a.rel_err, a.tier, a.min_frames, st)
    else:
        fx, fc = fuse(xyz, rgb, w, frame, voxel, a.tier, a.min_frames, st)
    n_cand, n_fused = len(xyz), len(fx)
    if n_fused > a.max_points:
        idx = np.random.default_rng(0).choice(n_fused, a.max_points, replace=False)
        fx, fc = fx[idx], fc[idx]
    text = report(st, n_cand, n_fused, len(fx), a.max_points)
    print(text, flush=True)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "points.ply", fx, fc.clip(0, 255).astype(np.uint8))
    sib = pathlib.Path(a.pred).parent.parent / "poses.npz"
    fidx = np.load(sib)["frame_idx"] if sib.exists() else np.arange(n)
    np.savez(out / "poses.npz", poses=poses.astype(np.float32), frame_idx=fidx)
    stats = {"frames": n, "voxel": voxel, "candidates": int(n_cand), "fused_points": int(n_fused),
             "written": int(len(fx)), "stages": {k: int(v) for k, v in st.items()},
             "params": {k: v for k, v in args.items() if k not in ("pred", "out")}, "per_frame_candidates": kept}
    (out / "fusion_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[fuse] wrote {len(fx):,} points -> {out}", flush=True)


if __name__ == "__main__":
    main()
