"""Camera height above the road from the track gauge, in every frame (no focal length, no pitch needed).

    python scripts/rail_height.py --video outputs/SF_1906_4k_gc/SF_1906_4k_gc.mp4 --start 1580 --end 10500 --step 4 \\
        --init "595,1174,865,440 1160,1174,865,440" --out data/rail_height.csv

The two rails the camera car runs on are parallel lines on the road. In the image each is a straight line,
x = a + b y, and both meet at the road's vanishing point. On a road plane, at any image row, the horizontal gap
between the rails is (gauge / camera height) times the row's distance below that vanishing point (pinhole camera,
any focal length, any pitch; roll and yaw change it only by their cosines). So

    camera height = effective gauge / (b_right - b_left)

where b are the rails' slopes dx/dy. The rails are found as bright strips (rail heads catch the sky): the strip
centres are measured, so the effective gauge is the gauge plus one rail-head width (--gauge-mm + --head-mm).

Tracking: each frame starts from the previous frame's lines (--init gives two lines as x0,y0,x1,y1 in the first
frame). In each row between --rows the strip nearest the predicted position is found with a ridge filter whose width
follows the expected head width; then each rail is a robust (RANSAC) line fit to its rows. A frame is ok when both
rails have enough inlier rows over a long enough span; otherwise the previous lines are kept as the prediction.

CSV: frame, aL, bL, aR, bR, nL, nR (inlier rows), rmsL, rmsR (px), vp_x, vp_y, h_over_g (camera height in effective
gauges), h_m, ok.
"""
import argparse
import csv
import pathlib
import subprocess

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def frames(video, start, end, step, size):
    w, h = size
    sel = f"between(n\\,{start}\\,{end})*not(mod(n-{start}\\,{step}))"
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{sel}'", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = start
    while True:
        buf = p.stdout.read(w * h)
        if len(buf) < w * h:
            break
        yield n, np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
        n += step
    p.wait()


def video_size(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def ridge_x(row, xc, half_w, search):
    """Sub-pixel x of the brightest strip of half-width half_w within `search` px of xc; (x, strength) or None."""
    n = len(row)
    w = max(2, int(round(half_w)))
    lo, hi = int(xc - search - 3 * w), int(xc + search + 3 * w) + 1
    if lo < 0 or hi > n:
        return None
    seg = row[lo:hi]
    core = uniform_filter1d(seg, 2 * w + 1, mode="nearest")
    flank = 0.5 * (np.roll(core, 2 * w + 1) + np.roll(core, -(2 * w + 1)))
    r = core - flank
    i0, i1 = 3 * w, len(seg) - 3 * w
    k = i0 + int(np.argmax(r[i0:i1]))
    if k <= i0 or k >= i1 - 1:
        return None
    a, b, c = r[k - 1], r[k], r[k + 1]
    d = a - 2 * b + c
    off = 0.5 * (a - c) / d if d < 0 else 0.0
    noise = np.std(r[i0:i1]) + 1e-6
    return lo + k + off, b / noise


def fit_line(ys, xs, thr, rng, iters=200):
    """RANSAC line x = a + b y; returns (a, b, inlier mask) or None."""
    n = len(ys)
    if n < 6:
        return None
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if ys[i] == ys[j]:
            continue
        b = (xs[j] - xs[i]) / (ys[j] - ys[i])
        a = xs[i] - b * ys[i]
        inl = np.abs(xs - (a + b * ys)) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 6:
        return None
    for _ in range(2):
        b, a = np.polyfit(ys[best], xs[best], 1)
        best = np.abs(xs - (a + b * ys)) < thr
    b, a = np.polyfit(ys[best], xs[best], 1)
    return a, b, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--init", required=True, help='"xL0,yL0,xL1,yL1 xR0,yR0,xR1,yR1": the two rails in the first frame')
    ap.add_argument("--rows", type=int, nargs=2, default=[700, 1145], help="image rows used (near the camera)")
    ap.add_argument("--row-step", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=150, help="ignore rows where the rails are closer than this (px)")
    ap.add_argument("--gauge-mm", type=float, default=1435.1, help="track gauge (4 ft 8.5 in)")
    ap.add_argument("--head-mm", type=float, default=60.0, help="rail-head width (strip centres are measured)")
    ap.add_argument("--min-strength", type=float, default=1.3, help="ridge peak / spread in the search window")
    ap.add_argument("--min-rows", type=int, default=25, help="inlier rows per rail for an ok frame")
    ap.add_argument("--min-span", type=float, default=150, help="vertical span of inliers per rail (px)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    size = video_size(a.video)
    lines = []
    for s in a.init.split():
        x0, y0, x1, y1 = map(float, s.split(","))
        b = (x1 - x0) / (y1 - y0)
        lines.append([x0 - b * y0, b])
    pred = np.array(lines)  # [[aL, bL], [aR, bR]]
    geff = (a.gauge_mm + a.head_mm) / 1000.0
    head_frac = a.head_mm / (a.gauge_mm + a.head_mm)
    rng = np.random.default_rng(0)
    ys_all = np.arange(a.rows[0], a.rows[1] + 1, a.row_step)
    rows_out = []
    n_ok = 0
    for n, img in frames(a.video, a.start, a.end, a.step, size):
        sm = gaussian_filter1d(img, 1.2, axis=1)
        res = []
        for side in (0, 1):
            ys, xs = [], []
            for y in ys_all:
                xl, xr = pred[0, 0] + pred[0, 1] * y, pred[1, 0] + pred[1, 1] * y
                gap = xr - xl
                if gap < a.min_gap:
                    continue
                xc = (xl, xr)[side]
                row = sm[y - 1:y + 2].mean(0)
                r = ridge_x(row, xc, 0.5 * head_frac * gap, max(8.0, 0.05 * gap))
                if r is not None and r[1] > a.min_strength:
                    ys.append(y)
                    xs.append(r[0])
            ys, xs = np.array(ys, float), np.array(xs, float)
            f = fit_line(ys, xs, 2.0, rng)
            if f is None:
                res.append(None)
                continue
            aa, bb, inl = f
            span = np.ptp(ys[inl]) if inl.any() else 0
            rms = float(np.sqrt(np.mean((xs[inl] - aa - bb * ys[inl]) ** 2)))
            res.append((aa, bb, int(inl.sum()), rms, span))
        ok = all(r is not None and r[2] >= a.min_rows and r[4] >= a.min_span for r in res)
        if ok:
            (aL, bL, nL, rL, _), (aR, bR, nR, rR, _) = res
            ok = bR - bL > 0.1
        if ok:
            vy = (aL - aR) / (bR - bL)
            vx = aL + bL * vy
            hg = 1.0 / (bR - bL)
            rows_out.append([n, aL, bL, aR, bR, nL, nR, rL, rR, vx, vy, hg, hg * geff, 1])
            # follow the rails, but slowly enough that one bad frame can't throw the tracker off
            pred = 0.3 * pred + 0.7 * np.array([[aL, bL], [aR, bR]])
            n_ok += 1
        else:
            rows_out.append([n] + [""] * 12 + [0])
        if n % 400 == a.start % 400:
            print(f"[rail] frame {n}: ok so far {n_ok}/{len(rows_out)}"
                  + (f", h = {rows_out[-1][12]:.2f} m" if ok else ""), flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "aL", "bL", "aR", "bR", "nL", "nR", "rmsL", "rmsR", "vp_x", "vp_y", "h_over_g", "h_m", "ok"])
        for r in rows_out:
            w.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in r])
    h = np.array([r[12] for r in rows_out if r[-1] == 1])
    print(f"[rail] {len(h)}/{len(rows_out)} frames ok; camera height median {np.median(h):.3f} m, "
          f"IQR {np.percentile(h, 25):.3f}-{np.percentile(h, 75):.3f} m (effective gauge {geff:.3f} m)")


if __name__ == "__main__":
    main()
