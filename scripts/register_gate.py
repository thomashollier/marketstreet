"""Register a full-width film scan to the camera gate, drop pulldown duplicates, crop a fixed picture area.

    python scripts/register_gate.py --input scan.mp4 --out reg.mp4 --start-sec 26 --end-sec 698

Why: a content-based stabiliser moves the image around to cancel shake, which moves the optical centre from
frame to frame; a reconstruction model then reads that as camera rotation. The edges of the picture area
(frame lines top and bottom, the gate edges left and right) are the shadow of the camera's aperture plate,
which is fixed to the lens. Lining every frame up on those edges gives a fixed optical centre, and leaves
every other motion (the real camera shake, the car's motion) alone.

How:
  pass 1  decode small (--det-width), keep a row profile of the vertical gradient (central columns: the frame
          lines) and a column profile of the horizontal gradient (central rows: the gate edges), plus a thumbnail.
          Frames identical to the previous one (24 fps pulldown of a 16 fps scan) are dropped.
          Each frame's x/y offset is found by 1D correlation of its profiles with a reference, which is then
          rebuilt from the aligned profiles (a few rounds).
  pass 2  decode at --out-width, shift each kept frame by its offset and cut the fixed crop (--crop, in
          fractions of the frame at the reference position), encode H.264 at --fps.
Writes <out>.npz with the kept source frame numbers, offsets (det-width pixels) and correlation scores,
and <out>_median.png (the aligned median frame with the crop drawn) for checking the crop.
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return s["width"], s["height"], float(num) / float(den)


def frames(path, w, h, start, end):
    """Yield gray uint8 frames [h, w] between start and end seconds."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    if end > start:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 8)
    n = w * h
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w)
    p.wait()


def profiles(g):
    g = g.astype(np.float32)
    h, w = g.shape
    ry = np.diff(g[:, int(.2 * w):int(.8 * w)], axis=0).mean(1)     # frame lines
    cx = np.diff(g[int(.2 * h):int(.8 * h), :], axis=1).mean(0)     # gate edges
    return ry, cx


def edge_mask(ref, halfwidth):
    """1 within `halfwidth` of the strongest edge in the first and last quarter of a reference profile
    (the two frame lines, or the two gate edges), 0 elsewhere, so picture content doesn't drive the match."""
    n = len(ref)
    m = np.zeros(n)
    for lo, hi in ((0, n // 4), (3 * n // 4, n)):
        k = lo + int(np.argmax(np.abs(ref[lo:hi])))
        m[max(0, k - halfwidth):k + halfwidth + 1] = 1
    return m


def best_shift(p, ref, search, mask=None):
    """Shift s (subpixel) that maximises corr(p[i + s], ref[i]) over the reference positions i in mask."""
    n = len(ref)
    m = np.ones(n) if mask is None else mask
    sh = np.arange(-search, search + 1)
    c = np.full(len(sh), -np.inf)
    for j, s in enumerate(sh):
        i = np.flatnonzero(m[max(0, -s):n - max(0, s)]) + max(0, -s)
        a, b = p[i + s], ref[i]
        a, b = a - a.mean(), b - b.mean()
        c[j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    k = int(np.argmax(c))
    d = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            d = 0.5 * (c[k - 1] - c[k + 1]) / den
    return sh[k] + d, float(c[k])


def shifted(p, s):
    x = np.arange(len(p))
    return np.interp(x + s, x, p)


def align(P, search, min_corr, masked, label, rounds=4):
    """Offset of each profile in P against a reference rebuilt from the aligned profiles each round.
    masked: after the first round, match only near the reference's two strongest edges (see edge_mask).
    Matches on the search limit or weaker than min_corr are filled from their neighbours (score 0);
    isolated spikes against a 7-frame running median are replaced by it (score negated)."""
    from scipy.ndimage import median_filter
    o = np.zeros(len(P))
    hw = max(4, P.shape[1] // 20)
    for it in range(rounds):
        ref = np.mean([shifted(p, s) for p, s in zip(P, o)], axis=0)
        m = edge_mask(ref, hw) if (masked and it) else None
        res = [best_shift(p, ref, search, m) for p in P]
        o = np.array([r[0] for r in res]); c = np.array([r[1] for r in res])
        bad = (np.abs(o) >= search - 1) | (c < min_corr)
        if bad.any() and (~bad).sum() > 1:
            o[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), o[~bad])
            c[bad] = 0.0
        o -= np.median(o)
    med = median_filter(o, size=7, mode="nearest")
    spike = np.abs(o - med) > max(2.0, 3 * 1.4826 * np.median(np.abs(o - med)))
    o[spike] = med[spike]
    c[spike] = -np.abs(c[spike])
    print(f"[gate] {label}: sd {o.std():.2f}px (range {o.min():+.1f}..{o.max():+.1f}), median score {np.median(c):.2f}, "
          f"filled {int((c == 0).sum())}, spikes repaired {int(spike.sum())}", flush=True)
    return o, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=-1.0)
    ap.add_argument("--det-width", type=int, default=640)
    ap.add_argument("--out-width", type=int, default=2048, help="decode width for the output (crop is cut from this)")
    ap.add_argument("--crop", type=float, nargs=4, default=[0.115, 0.11, 0.915, 0.895],
                    metavar=("X0", "Y0", "X1", "Y1"), help="fixed crop, fractions of the frame at the reference position")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--dup-thresh", type=float, default=0.25, help="drop a frame whose thumbnail change is below this x the median")
    ap.add_argument("--search", type=int, default=40, help="max offset searched, det-width pixels")
    ap.add_argument("--min-corr", type=float, default=0.3, help="weaker profile matches are treated as missing and filled from neighbours")
    ap.add_argument("--gate-smooth", type=float, default=2.0, help="seconds over which the frame-line/perforation offset is smoothed")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--analyse-only", action="store_true", help="pass 1 only (offsets + median image)")
    a = ap.parse_args()

    W, H, src_fps = probe(a.input)
    dw = a.det_width
    dh = int(round(H * dw / W / 2)) * 2
    print(f"[gate] {a.input}: {W}x{H} @ {src_fps:g} fps; detecting at {dw}x{dh}", flush=True)

    RY, CX, TH, ST = [], [], [], []
    sw = int(0.14 * dw)                     # perforation strips, left and right
    for g in frames(a.input, dw, dh, a.start_sec, a.end_sec):
        ry, cx = profiles(g)
        RY.append(ry); CX.append(cx)
        TH.append(g[::8, ::8].astype(np.float32))
        ST.append(np.concatenate([g[:, :sw], g[:, -sw:]], axis=1))
        if len(RY) % 2000 == 0:
            print(f"[gate] pass 1: {len(RY)} frames", flush=True)
    RY, CX, TH = np.array(RY), np.array(CX), np.array(TH)
    n = len(RY)
    diff = np.r_[np.inf, np.abs(np.diff(TH, axis=0)).mean((1, 2))]
    keep = diff > a.dup_thresh * np.median(diff[1:])
    idx = np.flatnonzero(keep)
    print(f"[gate] {n} decoded frames, {len(idx)} kept after dropping {n - len(idx)} duplicates", flush=True)

    # x: the gate edges.  y: frame-to-frame weave from the perforations (always high contrast), plus the
    # slowly varying offset between perforations and frame lines (the gate) from the frame lines, smoothed
    # over --gate-smooth seconds, so faint frame lines in some frames can't throw a frame off.
    from scipy.ndimage import median_filter
    ox, sx = align(CX[idx], a.search, a.min_corr, True, "x (gate edges)")
    oy_line, sy = align(RY[idx], a.search, a.min_corr, True, "y (frame lines)")
    ST = np.array(ST)[idx].astype(np.float32)
    # undo each frame's sideways shift (whole pixels is plenty for picking the perforation columns)
    strips = np.array([np.concatenate([np.roll(st[:, :sw], -int(round(dx)), axis=1),
                                       np.roll(st[:, sw:], -int(round(dx)), axis=1)], axis=1)
                       for st, dx in zip(ST, ox)])
    med = np.median(strips[::max(1, len(strips) // 300)], axis=0)  # [dh, 2*sw]
    perf_cols = np.std(med, axis=0) >= np.percentile(np.std(med, axis=0), 75)
    perf_prof = strips[:, :, perf_cols].mean(2)                      # [n, dh] brightness down the perf columns
    oy_perf, sp = align(perf_prof, min(a.search, dh // 16), a.min_corr, False, "y (perforations)")
    gate_minus_perf = median_filter(oy_line - oy_perf, size=max(3, int(a.gate_smooth * a.fps) | 1), mode="nearest")
    oy = oy_perf + gate_minus_perf
    print(f"[gate] frame line - perforation offset: range {gate_minus_perf.min():+.1f}..{gate_minus_perf.max():+.1f}px "
          f"(slow print/gate drift that perforations alone would miss)", flush=True)
    src_frame = np.round(a.start_sec * src_fps).astype(int) + idx
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), src_frame=src_frame, offset_x=ox, offset_y=oy, corr_x=sx, corr_y=sy,
             offset_y_frameline=oy_line, offset_y_perf=oy_perf, corr_perf=sp,
             det_size=np.array([dw, dh]), crop=np.array(a.crop))
    print(f"[gate] final: y sd {oy.std():.2f}px (range {oy.min():+.1f}..{oy.max():+.1f}), "
          f"x sd {ox.std():.2f}px (range {ox.min():+.1f}..{ox.max():+.1f}), "
          f"frame-to-frame |dy| 99% {np.percentile(np.abs(np.diff(oy)), 99):.1f}px", flush=True)

    # aligned median of a sample of frames, crop drawn, to check the crop
    import cv2
    pick = set(idx[np.linspace(0, len(idx) - 1, min(200, len(idx))).astype(int)].tolist())
    stack = []
    for i, g in enumerate(frames(a.input, dw, dh, a.start_sec, a.end_sec)):
        if i in pick:
            k = int(np.searchsorted(idx, i))
            M = np.float32([[1, 0, -ox[k]], [0, 1, -oy[k]]])
            stack.append(cv2.warpAffine(g, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    med = cv2.cvtColor(np.median(np.array(stack), axis=0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = (np.array(a.crop) * [dw, dh, dw, dh]).astype(int)
    cv2.rectangle(med, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.imwrite(str(out.with_name(out.stem + "_median.png")), med)
    if a.analyse_only:
        return

    ow = a.out_width
    oh = int(round(H * ow / W / 2)) * 2
    s = ow / dw
    cx0, cy0, cx1, cy1 = a.crop[0] * ow, a.crop[1] * oh, a.crop[2] * ow, a.crop[3] * oh
    cw, ch = int(round((cx1 - cx0) / 2)) * 2, int(round((cy1 - cy0) / 2)) * 2
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{cw}x{ch}", "-r", f"{a.fps:g}", "-i", "-",
                            "-c:v", "libx264", "-crf", str(a.crf), "-preset", "slow", "-pix_fmt", "yuv420p", str(out)],
                           stdin=subprocess.PIPE)
    keepset = {int(i): k for k, i in enumerate(idx)}
    for i, g in enumerate(frames(a.input, ow, oh, a.start_sec, a.end_sec)):
        k = keepset.get(i)
        if k is None:
            continue
        # output pixel (u, v) <- source pixel (u + crop_x0 + offset_x, v + crop_y0 + offset_y)
        M = np.float32([[1, 0, -(cx0 + ox[k] * s)], [0, 1, -(cy0 + oy[k] * s)]])
        enc.stdin.write(cv2.warpAffine(g, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).tobytes())
        if k % 2000 == 0:
            print(f"[gate] pass 2: {k}/{len(idx)}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"[gate] wrote {out}: {len(idx)} frames {cw}x{ch} @ {a.fps:g} fps", flush=True)


if __name__ == "__main__":
    main()
