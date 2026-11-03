"""Lens distortion from lines that should be straight (plumb-line calibration).

    python scripts/lens_lines.py outputs/<movie>/sfm_pilot/images --focal 3332

Rails, kerbs, cornices, window rows and building edges are straight in the world, so they must be straight in an
undistorted image. In every frame: line segments (OpenCV LSD) are joined into long lines (pieces that are nearly
collinear and close end to end, since film grain breaks edges up), and points are sampled along each line. Then
the radial distortion is found that makes all of those lines straightest at once, with the distortion centre at
the image centre (gate-registered frames: the optical centre is fixed).

Distortion follows COLMAP's convention so the result can be compared with / fed to sfm_colmap.py:
    distorted = undistorted * (1 + k1 r^2 + k2 r^4),  r = |undistorted - centre| / focal
Lines only see the distortion in pixels, i.e. k1 / focal^2 (and k2 / focal^4): pass the focal you'll use (--focal)
to get k1 in COLMAP terms; the pixel displacement at the corners is printed too, which needs no focal.

Checks: a held-out half of the frames (the fit must straighten lines it never saw), and --self-test, which
distorts the frames by a known k1 and checks that it is recovered.
"""
import argparse
import pathlib

import cv2
import numpy as np


def segments(gray, seg_min=25):
    """LSD segments [n, 4] (x0, y0, x1, y1) longer than seg_min px."""
    segs = np.asarray(cv2.createLineSegmentDetector().detect(gray)[0]).reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[L > seg_min]


def long_lines(raw, und, min_len=250, ang_tol=1.5, dist_tol=2.0, gap=60):
    """Join segments into long lines. Joining is done on `und` (the segments undistorted with the current
    estimate, so lines the lens bends can still be joined); the returned points come from `raw` (as seen),
    sampled every ~8 px along each member segment: a list of [n, 2] arrays."""
    segs = und
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % np.pi
    order = np.argsort(-L)
    used = np.zeros(len(segs), bool)
    out = []
    for i in order:
        if used[i]:
            continue
        group = [i]
        used[i] = True
        grew = True
        while grew:                      # grow along the current line fit
            grew = False
            P = np.concatenate([segs[group, :2], segs[group, 2:]])
            c = P.mean(0)
            d = np.linalg.svd(P - c)[2][0]
            nrm = np.array([-d[1], d[0]])
            t = (P - c) @ d
            lo, hi = t.min(), t.max()
            a0 = np.arctan2(d[1], d[0]) % np.pi
            cand = np.flatnonzero(~used & (np.minimum(np.abs(ang - a0), np.pi - np.abs(ang - a0)) < np.radians(ang_tol)))
            for j in cand:
                e = segs[j].reshape(2, 2)
                if np.abs((e - c) @ nrm).max() > dist_tol:
                    continue
                tj = (e - c) @ d
                if tj.max() < lo - gap or tj.min() > hi + gap:
                    continue
                group.append(j)
                used[j] = True
                grew = True
        P = np.concatenate([segs[group, :2], segs[group, 2:]])
        c = P.mean(0)
        d = np.linalg.svd(P - c)[2][0]
        t = (P - c) @ d
        if t.max() - t.min() < min_len or len(group) < 2:
            continue
        pts = []
        for j in group:                  # sample each member segment (as seen) every ~8 px
            n = max(2, int(L[j] // 8))
            s = np.linspace(0, 1, n)[:, None]
            pts.append(raw[j, :2] * (1 - s) + raw[j, 2:] * s)
        out.append(np.concatenate(pts))
    return out


def undistort(p, c, f, k1, k2):
    """Inverse of distorted = u * (1 + k1 r^2 + k2 r^4) (fixed-point iterations), pixels in and out."""
    x = (p - c) / f
    u = x.copy()
    for _ in range(20):
        r2 = (u ** 2).sum(1, keepdims=True)
        u = x / (1 + k1 * r2 + k2 * r2 ** 2)
    return u * f + c


def straightness(lines, c, f, k1, k2):
    """RMS distance (px) of the points of each line from its best straight line, after undistortion."""
    r = []
    for P in lines:
        U = undistort(P, c, f, k1, k2)
        U = U - U.mean(0)
        r.append(np.linalg.svd(U, compute_uv=False)[1] / np.sqrt(len(U)))
    return np.array(r)


def fit(lines, c, f, two_terms, trim=0.1):
    """k1 (and k2) minimising the summed squared straightness residual; worst `trim` of lines dropped
    (curved things, mis-joined lines) after a first pass."""
    from scipy.optimize import minimize
    def cost(x, sel):
        k1, k2 = x[0], (x[1] if two_terms else 0.0)
        return (straightness([lines[i] for i in sel], c, f, k1, k2) ** 2).sum()
    sel = np.arange(len(lines))
    x0 = np.zeros(2 if two_terms else 1)
    for _ in range(2):
        x = minimize(cost, x0, args=(sel,), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6}).x
        r = straightness(lines, c, f, x[0], x[1] if two_terms else 0.0)
        sel = np.flatnonzero(r <= np.quantile(r, 1 - trim))
        x0 = x
    return x[0], (x[1] if two_terms else 0.0), sel


def corner_shift(c, f, k1, k2, w, h):
    """How far (px) distortion moves the image corner: the size of the effect, independent of the focal."""
    corner = np.array([[w, h]], float)
    u = undistort(corner, c, f, k1, k2)
    return float(np.hypot(*(corner - u)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder of frames")
    ap.add_argument("--focal", type=float, required=True, help="focal length in pixels at this image size (for k1 in COLMAP terms)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--min-len", type=float, default=0.15, help="shortest line kept, fraction of image width")
    ap.add_argument("--two-terms", action="store_true", help="fit k1 and k2")
    ap.add_argument("--self-test", type=float, default=None, metavar="K1", help="distort the frames by this k1 first and check it is recovered")
    a = ap.parse_args()

    files = sorted(pathlib.Path(a.images).glob("*.jpg"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(a.max_frames, len(files))).astype(int)]
    h, w = cv2.imread(str(files[0]), 0).shape
    c = np.array([w / 2, h / 2])
    SEG = []
    for fp in files:
        g = cv2.imread(str(fp), 0)
        if a.self_test is not None:                       # resample the frame through a known distortion
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            src = undistort(np.c_[xx.ravel(), yy.ravel()], c, a.focal, a.self_test, 0.0).reshape(h, w, 2).astype(np.float32)
            g = cv2.remap(g, src[..., 0], src[..., 1], cv2.INTER_CUBIC)
        SEG.append(segments(g))

    # join lines with the current estimate undone, fit, repeat: lines the lens bends get joined once it's known
    k1 = k2 = 0.0
    for it in range(4):
        lines_a, lines_b = [], []
        for n, sg in enumerate(SEG):
            u = np.c_[undistort(sg[:, :2], c, a.focal, k1, k2), undistort(sg[:, 2:], c, a.focal, k1, k2)]
            (lines_a if n % 2 == 0 else lines_b).extend(long_lines(sg, u, min_len=a.min_len * w))
        k1, k2, sel = fit(lines_a, c, a.focal, a.two_terms)
        print(f"  round {it}: {len(lines_a) + len(lines_b)} long lines, k1 {k1:+.4f}" + (f" k2 {k2:+.4f}" if a.two_terms else ""))
    lengths = np.array([np.ptp(P @ np.linalg.svd(P - P.mean(0))[2][0]) for P in lines_a + lines_b])
    print(f"{len(files)} frames {w}x{h}: {len(lines_a) + len(lines_b)} long lines "
          f"(median length {np.median(lengths):.0f}px, longest {lengths.max():.0f}px)")
    r0_a = straightness([lines_a[i] for i in sel], c, a.focal, 0, 0)
    r1_a = straightness([lines_a[i] for i in sel], c, a.focal, k1, k2)
    r0_b = straightness(lines_b, c, a.focal, 0, 0)
    r1_b = straightness(lines_b, c, a.focal, k1, k2)
    keep_b = r0_b <= np.quantile(r0_b, 0.9)
    print(f"fit on half the frames ({len(sel)} lines): k1 = {k1:+.4f}" + (f", k2 = {k2:+.4f}" if a.two_terms else "") +
          f" at focal {a.focal:.0f}px  (k1/f^2 = {k1 / a.focal ** 2:+.3e} per px^2)")
    print(f"  corner of the image moved {corner_shift(c, a.focal, k1, k2, w, h):.2f}px by this distortion "
          f"({'barrel' if k1 < 0 else 'pincushion'})")
    print(f"  line straightness, fitted half:   {np.sqrt((r0_a ** 2).mean()):.3f}px -> {np.sqrt((r1_a ** 2).mean()):.3f}px RMS")
    print(f"  line straightness, held-out half: {np.sqrt((r0_b[keep_b] ** 2).mean()):.3f}px -> {np.sqrt((r1_b[keep_b] ** 2).mean()):.3f}px RMS")
    if a.self_test is not None:
        print(f"  self-test: applied k1 = {a.self_test:+.4f}, recovered {k1:+.4f}")


if __name__ == "__main__":
    main()
