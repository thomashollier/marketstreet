"""Focal length from known sizes: track gauge, street width, Ferry tower height, placed on the map.

    python scripts/lens_from_landmarks.py --model outputs/SF_1906_4k_gc/sfm_global/fixed_lens \\
        --rails data/rail_height.csv --tower data/ferry_tower_top.csv --profile data/market_street_profile.csv

Angles in an image only give focal length / distance, so a metric distance along the street is needed. Here it
comes from two places the SfM model and the map share (--fix SFM_ALONG:MAP_METRES): the Call Building (tallest
south-side points in the model; 3rd St + half the street + half the building on the map) and the Ferry Building
facade (the wall of points ending the model). SFM_ALONG is measured from the first camera along the path.

Three measurements of the focal length, each with a different known size:
  1. Ferry tower top (--tower-m, 245 ft): per frame, the angle between the tracked top and the horizon. The horizon
     is the rails' vanishing point (rail_height.py) moved by the road grade there (USGS profile), the rise is
     tower height + tower ground - local ground - camera height, the distance is from the map. Solved per frame.
  2. Camera height: rail_height.py gives it in metres from the gauge (no focal needed); the model's camera height
     above its own road, in metres via the map scale, is squeezed by f_true / f_model (with a wrong focal, sizes
     across the view scale with it, distances along the view don't). So f_true = f_model * h_model / h_rails.
  3. Street width: the model's facade-to-facade width against --street-m (120 ft between building lines), same
     squeeze.
"""
import argparse
import csv

import numpy as np


def street_frame(model):
    import pycolmap
    d = np.load(f"{model}/poses.npz")
    T, fi = d["poses"], d["frame_idx"]
    C = T[:, :3, 3]
    _, _, vt = np.linalg.svd(C - C.mean(0))
    ax = vt[0] if vt[0] @ (C[-1] - C[0]) > 0 else -vt[0]
    down = T[:, :3, 1].mean(0)
    down -= ax * (down @ ax)
    down /= np.linalg.norm(down)
    right = np.cross(down, ax)
    rec = pycolmap.Reconstruction(model)
    X = np.array([p.xyz for p in rec.points3D.values()])
    f_model = rec.cameras[next(iter(rec.cameras))].params[0] / rec.cameras[next(iter(rec.cameras))].width
    o = C[0]
    return dict(fi=fi, sig=(C - o) @ ax, cam_down=(C - o) @ down, a=(X - o) @ ax, r=(X - o) @ right,
                v=(X - o) @ down, f_model=f_model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rails", required=True)
    ap.add_argument("--tower", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--video-size", type=int, nargs=2, default=[1618, 1174], help="pixel size of the two CSVs")
    ap.add_argument("--fix", action="append", default=None, help="SFM_ALONG:MAP_METRES (two or more)")
    ap.add_argument("--tower-along", type=float, default=2645.0, help="Ferry tower along the profile (m)")
    ap.add_argument("--tower-ground", type=float, default=3.1)
    ap.add_argument("--tower-m", type=float, default=74.7, help="tracked tower top above its ground (245 ft)")
    ap.add_argument("--street-m", type=float, default=36.58, help="width between building lines (120 ft)")
    a = ap.parse_args()
    fixes = np.array([[float(x) for x in s.split(":")] for s in (a.fix or ["4.47:1425", "13.15:2640"])])

    k, x0 = np.polyfit(fixes[:, 0], fixes[:, 1], 1)
    print(f"[map] {k:.1f} m per model unit; first camera at {x0:.0f} m along the profile")
    m = street_frame(a.model)
    W, H = a.video_size
    cx, cy = W / 2, H / 2
    P = np.array([[float(r["distance_m"]), float(r["elevation_m"])] for r in csv.DictReader(open(a.profile))])
    R = {int(r["frame"]): r for r in csv.DictReader(open(a.rails)) if r["ok"] == "1"}
    T = {int(r["frame"]): r for r in csv.DictReader(open(a.tower)) if r["ok"] == "1"}
    sg = dict(zip(m["fi"].tolist(), m["sig"]))
    h_rail = float(np.median([float(r["h_m"]) for r in R.values()]))
    print(f"[rails] camera height {h_rail:.3f} m (median of {len(R)} frames)")

    # 1. tower top
    F = sorted(set(R) & set(T) & set(sg))
    x = np.array([x0 + k * sg[f] for f in F])
    vp = np.array([float(R[f]["vp_y"]) for f in F])
    vv = np.array([float(T[f]["v"]) for f in F])
    g = np.interp(x, P[:, 0], P[:, 1])
    grade = (np.interp(x + 20, P[:, 0], P[:, 1]) - np.interp(x + 2, P[:, 0], P[:, 1])) / 18
    want = np.arctan((a.tower_m + a.tower_ground - g - h_rail) / (a.tower_along - x))
    fs = np.empty(len(F))
    for i in range(len(F)):
        lo, hi = 0.5 * W, 6.0 * W
        for _ in range(50):
            fm = 0.5 * (lo + hi)
            vh = vp[i] + fm * grade[i]  # rails' vanishing point sits above the horizon by the grade ahead
            if np.arctan((cy - vv[i]) / fm) - np.arctan((cy - vh) / fm) > want[i]:
                lo = fm
            else:
                hi = fm
        fs[i] = fm
    f1 = np.median(fs)
    thirds = np.array_split(np.argsort(x), 3)
    print(f"[1 tower top]  f = {f1:.0f} px (IQR {np.percentile(fs, 25):.0f}-{np.percentile(fs, 75):.0f}, {len(F)} frames;"
          f" by thirds of the ride {', '.join(f'{np.median(fs[t]):.0f}' for t in thirds)})")

    # 2. camera height in the model vs rails
    f_model = m["f_model"] * W
    hs = []
    for i in range(0, len(m["sig"]), 10):
        sel = (np.abs(m["a"] - m["sig"][i]) < 0.03) & (np.abs(m["r"]) < 0.03) & (m["v"] > m["cam_down"][i])
        if sel.sum() > 20:
            hs.append(np.median(m["v"][sel]) - m["cam_down"][i])
    h_model = float(np.median(hs)) * k
    f2 = f_model * h_model / h_rail
    print(f"[2 camera height] model {h_model:.2f} m vs rails {h_rail:.2f} m -> f = {f2:.0f} px")

    # 3. street width: facade lines = densest lateral positions on each side, between 1 and 10 camera heights up
    up = -m["v"] * k
    sel = (up > 3) & (up < 20)
    widths = []
    for side in (1, -1):
        rr = side * m["r"][sel] * k
        hh, e = np.histogram(rr[(rr > 5) & (rr < 40)], np.arange(5, 40, 0.25))
        widths.append(e[np.argmax(hh)] + 0.125)
    w_model = sum(widths)
    f3 = f_model * w_model / a.street_m
    print(f"[3 street width] model {widths[0]:.1f} + {widths[1]:.1f} = {w_model:.1f} m vs {a.street_m:.2f} m -> f = {f3:.0f} px")

    fl = np.array([f1, f2, f3])
    print(f"[lens] model focal {f_model:.0f} px; measured {', '.join(f'{v:.0f}' for v in fl)} -> mean {fl.mean():.0f} px"
          f" (spread {fl.std() / fl.mean() * 100:.1f}%), hFOV {2 * np.degrees(np.arctan(W / 2 / fl.mean())):.1f} deg,"
          f" focal / width {fl.mean() / W:.3f}")


if __name__ == "__main__":
    main()
