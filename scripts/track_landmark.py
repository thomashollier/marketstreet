"""Find one landmark point (e.g. the top of the Ferry Building tower) in every frame, from hand-placed anchors.

    python scripts/track_landmark.py --video reg.mp4 --start 1580 --end 10500 --step 4 \\
        --anchors "1708:833,287,20 3308:842,353,25 4908:850,295,31" --out data/ferry_tower_top.csv

Each anchor is frame:u,v,T: the point's pixel position in that frame and a template half-width T in pixels (the
landmark's size there; T should scale with 1/distance). The template is the anchor frame around the point: T either
side, 0.5 T above and 2 T below (for a tower top: the spire and the tower body under it, which has texture).

Every frame is matched independently (no template updating, so errors can't accumulate) against the two anchors on
either side of it, with the template resized to the expected size (sizes interpolated on a log scale between the
anchors, refined over +-8%), normalised cross-correlation on lightly blurred images (grain). The point is the
score-weighted mean of the two; their disagreement is the error estimate. Outside the first/last anchor only the
nearest anchor is used. Position search is centred on the previous frame's result (falls back to the anchors).

CSV: frame, u, v (pixels in --video), ncc (best score), scale (size relative to the first anchor), disagree_px
(between the two anchors' answers; blank with one anchor), ok (score >= --min-ncc, disagreement <= --max-disagree,
inside the frame).
"""
import argparse
import pathlib
import subprocess

import cv2
import numpy as np


def read_frames(video, frames, box):
    """Gray crops (box = x0, y0, w, h) of evenly spaced frame numbers (range(start, end + 1, step)), in order."""
    x0, y0, w, h = box
    want = np.asarray(frames)
    f0, step = int(want[0]), int(want[1] - want[0]) if len(want) > 1 else 1
    sel = f"between(n\\,{f0}\\,{int(want[-1])})*not(mod(n-{f0}\\,{step}))"
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{sel}',crop={w}:{h}:{x0}:{y0}",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    out = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    if len(out) != len(want):
        raise RuntimeError(f"got {len(out)} frames, wanted {len(want)}")
    return out


def template(img, p, T):
    """Patch around p: T either side, 0.5T above, 2T below (clipped to the image). Returns patch, p's offset in it."""
    x0, x1 = int(round(p[0] - T)), int(round(p[0] + T)) + 1
    y0, y1 = int(round(p[1] - 0.5 * T)), int(round(p[1] + 2 * T)) + 1
    x0c, y0c = max(0, x0), max(0, y0)
    t = img[y0c:min(y1, img.shape[0]), x0c:min(x1, img.shape[1])]
    return t, np.array([p[0] - x0c, p[1] - y0c])


def match(img, tmpl, off, s, centre, search):
    """Template scaled by s, searched within `search` px of `centre` (where the point should be).
    Returns (point position, score)."""
    t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    o = off * s
    th, tw = t.shape
    x0, y0 = int(round(centre[0] - o[0] - search)), int(round(centre[1] - o[1] - search))
    x0c, y0c = max(0, x0), max(0, y0)
    win = img[y0c:y0 + th + 2 * search, x0c:x0 + tw + 2 * search]
    if win.shape[0] < th + 3 or win.shape[1] < tw + 3:
        return None, -1.0
    r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
    j, i = np.unravel_index(np.argmax(r), r.shape)
    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d
    dy = sub(r[j - 1, i], r[j, i], r[j + 1, i]) if 0 < j < r.shape[0] - 1 else 0.0
    dx = sub(r[j, i - 1], r[j, i], r[j, i + 1]) if 0 < i < r.shape[1] - 1 else 0.0
    return np.array([x0c + i + dx + o[0], y0c + j + dy + o[1]]), float(r[j, i])


def best_over_scales(img, tmpl, off, s0, centre, search, spread=0.08, n=9):
    best = (None, -1.0, s0)
    for s in s0 * np.linspace(1 - spread, 1 + spread, n):
        p, sc = match(img, tmpl, off, s, centre, search)
        if p is not None and sc > best[1]:
            best = (p, sc, s)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--anchors", required=True, help='"frame:u,v,T ..." hand-placed positions and template half-widths')
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--search", type=int, default=25, help="search radius around the previous frame's position, pixels")
    ap.add_argument("--blur", type=float, default=1.0, help="Gaussian sigma applied before matching (grain)")
    ap.add_argument("--box", type=int, nargs=4, default=None, metavar=("X0", "Y0", "W", "H"),
                    help="only decode this region (pixels); default: around the anchors")
    ap.add_argument("--margin", type=int, default=5, help="point must be at least this far inside the frame")
    ap.add_argument("--min-ncc", type=float, default=0.6)
    ap.add_argument("--max-disagree", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    anchors = {}
    for tok in a.anchors.split():
        f, vals = tok.split(":")
        u, v, T = (float(x) for x in vals.split(","))
        anchors[int(f)] = (np.array([u, v]), T)
    frames = list(range(a.start, a.end + 1, a.step))
    for f in anchors:
        if f not in frames:
            raise SystemExit(f"anchor frame {f} is not on the --start/--step grid")
    A = np.array([p for p, _ in anchors.values()])
    Tmax = max(T for _, T in anchors.values())
    box = a.box or [int(A[:, 0].min() - 4 * Tmax), 0, int(A[:, 0].max() - A[:, 0].min() + 8 * Tmax), int(A[:, 1].max() + 4 * Tmax)]
    box[0] = max(0, box[0])
    box = [b // 2 * 2 for b in box]          # the 4:2:0 video only crops to even sizes and offsets
    print(f"[track] {len(frames)} frames, decoding box {box}", flush=True)
    cache = pathlib.Path(a.out).with_suffix(f".frames_{a.start}_{a.end}_{a.step}_{'_'.join(map(str, box))}.npy")
    if cache.exists():
        F = np.load(cache, mmap_mode="r")
    else:
        F = read_frames(a.video, frames, box)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, F)
    off = np.array(box[:2], float)
    blur = lambda g: cv2.GaussianBlur(g.astype(np.float32), (0, 0), a.blur) if a.blur > 0 else g.astype(np.float32)

    idx = {f: k for k, f in enumerate(frames)}
    ak = sorted(idx[f] for f in anchors)
    T0 = anchors[frames[ak[0]]][1]
    tm = {}
    for k in ak:
        p, T = anchors[frames[k]]
        t, o = template(blur(F[k]), p - off, T)
        tm[k] = (t, o, T / T0, p - off)
    logS = np.interp(np.arange(len(frames)), ak, [np.log(tm[k][2]) for k in ak])   # expected size, log-interpolated

    P = np.full((len(frames), 2), np.nan)
    NCC = np.full(len(frames), np.nan)
    SC = np.full(len(frames), np.nan)
    DIS = np.full(len(frames), np.nan)
    order = list(range(ak[0], len(frames))) + list(range(ak[0] - 1, -1, -1))
    prev = None
    for n, k in enumerate(order):
        if k == ak[0] - 1:
            prev = tm[ak[0]][3]                       # restart backwards from the first anchor
        j = int(np.searchsorted(ak, k))
        near = [ak[j - 1], ak[j]] if 0 < j < len(ak) and ak[j] != k else [ak[min(j, len(ak) - 1)]]
        if k in tm:
            near = [k]
        img = blur(F[k])
        res = []
        for ka in near:
            t, o, Sa, pa = tm[ka]
            centre = prev if prev is not None else pa
            p, sc, s = best_over_scales(img, t, o, np.exp(logS[k]) / Sa, centre, a.search)
            if p is not None:
                res.append((p, sc, s * Sa))
        if not res:
            continue
        w = np.array([max(r[1], 1e-3) for r in res])
        P[k] = np.average([r[0] for r in res], axis=0, weights=w)
        NCC[k] = max(r[1] for r in res)
        SC[k] = np.average([r[2] for r in res], weights=w)
        if len(res) == 2:
            DIS[k] = np.linalg.norm(res[0][0] - res[1][0])
        if NCC[k] >= a.min_ncc and (np.isnan(DIS[k]) or DIS[k] <= a.max_disagree):
            prev = P[k]
        if n % 500 == 0:
            print(f"[track] {n}/{len(frames)} frames", flush=True)

    full = P + off
    vid_w, vid_h = (int(x) for x in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(a.video)], capture_output=True, text=True).stdout.strip().split(","))
    ok = (~np.isnan(P[:, 0]) & (NCC >= a.min_ncc) & (np.isnan(DIS) | (DIS <= a.max_disagree)) &
          (full[:, 0] > a.margin) & (full[:, 0] < vid_w - a.margin) & (full[:, 1] > a.margin) & (full[:, 1] < vid_h - a.margin))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("frame,u,v,ncc,scale,disagree_px,ok\n")
        for f, (u, v), s, sc, dis, o in zip(frames, full, NCC, SC, DIS, ok):
            fh.write(f"{f},{u:.2f},{v:.2f},{s:.3f},{sc:.3f},{'' if np.isnan(dis) else f'{dis:.2f}'},{int(o)}\n")
    two = ~np.isnan(DIS)
    print(f"[track] wrote {out}: {ok.sum()}/{len(frames)} frames usable; between anchors the two answers disagree by "
          f"{np.nanmedian(DIS):.2f}px median, {np.nanpercentile(DIS, 90):.2f}px 90th percentile "
          f"({(DIS[two] <= a.max_disagree).mean() * 100:.0f}% within {a.max_disagree}px)", flush=True)


if __name__ == "__main__":
    main()
