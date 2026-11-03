"""Ground elevation along a straight line between two points, from the USGS Elevation Point Query Service
(3DEP bare earth, US only), saved as CSV for check_level.py --profile.

    python scripts/usgs_profile.py --from 37.77838,-122.41459 --to 37.79540,-122.39370 --out data/market_street_profile.csv

Default endpoints: Market St at 8th St -> Market St at the Ferry Building (the 1906 ride).
CSV columns: distance_m (from --from), elevation_m, lat, lon.
"""
import argparse
import json
import math
import pathlib
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="a", default="37.77838,-122.41459", help="lat,lon")
    ap.add_argument("--to", dest="b", default="37.79540,-122.39370", help="lat,lon")
    ap.add_argument("--n", type=int, default=81, help="samples along the line")
    ap.add_argument("--out", default="data/market_street_profile.csv")
    a = ap.parse_args()
    (la0, lo0), (la1, lo1) = [tuple(map(float, s.split(","))) for s in (a.a, a.b)]
    R = 6371000.0
    L = R * math.hypot(math.radians(la1 - la0), math.radians(lo1 - lo0) * math.cos(math.radians((la0 + la1) / 2)))
    rows = []
    for i in range(a.n):
        t = i / (a.n - 1)
        lat, lon = la0 + t * (la1 - la0), lo0 + t * (lo1 - lo0)
        url = f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&wkid=4326&includeDate=false"
        for attempt in range(6):          # the service now and then answers with an empty body
            try:
                v = json.load(urllib.request.urlopen(url, timeout=60))["value"]
                break
            except (ValueError, OSError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        rows.append((t * L, float(v), lat, lon))
        print(f"{t * L:7.0f} m  {float(v):6.2f} m", flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("distance_m,elevation_m,lat,lon\n" + "".join(f"{d:.1f},{e:.2f},{la:.6f},{lo:.6f}\n" for d, e, la, lo in rows))
    print(f"wrote {out} ({L:.0f} m)")


if __name__ == "__main__":
    main()
