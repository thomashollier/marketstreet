# marketstreet

[LoGeR](https://github.com/Junyi42/LoGeR) (long-context geometric reconstruction) on [Modal](https://modal.com).
Movie in → camera poses + coloured point cloud out. Structure follows
[panosplat](https://github.com/thomashollier/panosplat): one app, one shared Volume, local entrypoints.

## Run it (from your machine, not Claude cloud)

```bash
pip install modal && modal setup          # once
modal run --detach modal_app.py::run --video ./clip.mp4
modal run modal_app.py::pull --movie clip # -> ./outputs/clip/LoGeR_star/{points.ply,poses.npz,trajectory.txt}
```

`run` uploads the movie to the `marketstreet-data` volume, downloads the checkpoint from Hugging Face
onto the volume (first run only), runs upstream `demo_viser.py --skip_viser` on an A100, then exports
`poses.npz` and `points.ply`. Useful flags: `--variant LoGeR|LoGeR_star`, `--stride`, `--start-frame`,
`--end-frame`, `--last-seconds N` (only the last N seconds; results go under `<stem>_last<N>s`), `--tag T` (results go under `<movie>/LoGeR_star_T`, so runs with different settings sit side by side; pull with `--variant LoGeR_star_T`), `--window-size`, `--overlap-size`, `--reset-every` (clips >1k frames).
For a very large movie, pre-upload with
`modal volume put marketstreet-data ./clip.mp4 /clip/input/clip.mp4` and pass `--skip-upload`.

## Keep the best observation of each surface (LoGeR)

Every pixel of every frame is a candidate surface point. `fuse` scores them (model confidence, depth, viewing angle), drops
flying pixels / grazing views / the farthest few percent per frame, then works nearest-first: a farther observation is dropped if a
better, nearer one already sits within its uncertainty (`--rel-err`, fraction of depth), so a distant view is replaced by the later
close-up but kept where nothing better exists. Surfaces seen by fewer than `--min-frames` distinct frames are dropped, which also
removes moving people and cars (they are in a new world position every frame). Camera poses are left untouched. CPU only, runs on
the saved raw predictions of a finished run:

```bash
modal run modal_app.py::fuse --movie <movie>                              # -> <movie>/LoGeR_star/fused/
modal run modal_app.py::pull --movie <movie> --variant LoGeR_star/fused
python scripts/align_to_path.py outputs/<movie>/LoGeR_star/fused --tumble
python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled outputs/<movie>/LoGeR_star/fused/aligned_tumbled
```

Use the same `--start-frame/--end-frame/--stride` as the original run. It prints (and saves in `fusion_stats.json`) how many
pixels each stage removed, so you can see what is thinning the cloud. The flying-pixel test allows for how fast depth changes on a
flat surface at that tilt (so the distant road and building faces survive); grazing views are down-weighted, not deleted.

Dials: more points = `--min-frames 1` (also keeps moving objects), `--conf-pct 10`, `--rtol 0.1`; cleaner = `--min-frames 3`,
`--rel-err 0.05`. On a synthetic street with distance-growing depth errors and a moving object, defaults give median error 0.039
vs 0.072 for a naive merge, bad points 2% vs 14%, floaters 0.05% vs 3%, similar coverage; `--min-frames 3` halves the points and
cuts bad points to 0.5%. Not yet tuned on real footage.

## Start from the original film scan (gate registration)

A content-based stabiliser moves each frame to cancel shake, which moves the optical centre; a reconstruction model
reads that as camera rotation and bends the street. Starting from a full-width scan (perforations and frame lines
visible) avoids it: the frame lines and gate edges are the shadow of the camera's aperture plate, fixed to the lens.

```bash
modal run --detach modal_app.py::fetch --movie SF_1906_4k \
    --url https://archive.org/download/MarketStreet19064KScan20181016/Market_Street_1906_4K_scan_20181016.hevc.mp4
modal run --detach modal_app.py::register --movie SF_1906_4k --video SF_1906_4k.mp4 --out-movie SF_1906_4k_gate \
    --start-sec 25.5 --end-sec 699
modal run --detach modal_app.py::run --video SF_1906_4k_gate.mp4 --skip-upload --stride 4
```

`fetch` downloads straight onto the volume (no local copy) and saves a few sample frames. `register`
(`scripts/register_gate.py`) drops the 24 fps pulldown duplicates (back to 16 fps), takes the frame-to-frame weave
from the perforations and the slow perforation-to-gate drift from the frame lines (smoothed over 2 s, so faint frame
lines can't throw a frame off), x from the gate edges, then cuts one fixed crop inside the picture area (`--crop`,
fractions of the frame; check `<out-movie>_median.png`). Offsets per frame go in `<out-movie>.npz`.

## Is the street level? (check for bowing)

```bash
python scripts/check_level.py outputs/<movie>/LoGeR_star      # also writes level_check.png next to the results
```

Measures, in a side view along the straight path, how much the camera height and the road surface "sag" compared with the
straight line joining their two ends, as a % of path length and in camera heights (under 0.5% = level; bowl = middle lower,
hill = middle higher), plus the change in ground slope from start to end. Nothing is a fixed distance: the camera's height above
the road is measured from the cloud, and the road search (`--band`, `--below`) is in camera heights, each point compared with the
camera at its own position along the street. `--trim-end 0.02` drops a turn at the end of the line. The plot is vertically
exaggerated. Tested on synthetic flat / hill / bowl streets at different scales (hill of 10 camera heights reads 9.6).

### Against the real street

Market Street is not level: the USGS bare-earth profile (`data/market_street_profile.csv`, from
`scripts/usgs_profile.py`) has a ~5 m crest about halfway between 8th St and the Ferry Building. Compare with it:

```bash
python scripts/check_level.py outputs/SF_1906_4k_gc/sfm_global27/fixed_lens \
    --profile data/market_street_profile.csv --profile-range 814 2609
```

`--profile-range` is where the first and last camera are along the profile (metres). The path is scaled to that
distance; it prints the camera height this implies (a check: the track gauge gives 1.97 m on the 1906 ride) and the road's height above the
line joining its ends against the real street's, in metres, and draws the real street in green.

## Lens, landmarks and structure from motion

- `scripts/sfm_colmap.py` (Modal: `modal run modal_app.py::sfm ...`): masked SIFT, sequential matching, COLMAP
  mapping with one shared camera; `fixed_lens/` and `free_lens/` results.
  For a long ride use `--mapper global`: on frames 1580-10500 of the 1906 scan the incremental mapper hinged where
  big moving objects filled the view (a 246 m "hill"); the global mapper on the same matches, with the measured
  lens, follows the USGS profile to 0.7 m RMS and the Ferry tower (not used in the solve) fits one 3D point to 1.9 px
  (90%; median 1.1 px, 1543 frames; measured on `sfm_global27/fixed_lens`, the same as the first global run's 1.9 px).
  So the model needs no tower correction: `landmark_ba.py` on it (`sfm_global27/landmark`) did not converge from its
  coarse start (4.2 px at 90%) and is not an improvement.
  The worst frames are 8000-9499 (90% ~37 px in both models), so that looks like the track, not the model. This is
  the point refined to its best fit; `landmark_ba.py`'s "before" line uses only its coarse starting point (6.9 px here).
- `scripts/lens_lines.py`: radial distortion from lines that should be straight (on the 4K scan: negligible).
  Cross-check by bundle adjustment on `sfm_global27` (every 4th frame, 60K points, lens held at 27.2 deg, k1 free):
  k1 = 0.014 (slight pincushion), 1.2 px at the corners and 0.7 px mid-edge, reprojection 0.96 -> 0.88 px. Freeing
  the focal too gives 25.3 deg with k1 = 0.038 at the same error, so the images can't separate the two: keep the
  measured lens and k1 = 0.
- `scripts/track_landmark.py`: a landmark point (the Ferry Building tower top, `data/ferry_tower_top.csv`) in every
  frame from a few hand-placed anchors.
- `scripts/landmark_ba.py`: adds that point to a COLMAP model as one 3D point seen in every frame and re-adjusts.
- `scripts/rail_height.py`: camera height above the road from the track gauge, every frame. The car's two rails
  meet at the road's vanishing point; camera height = effective gauge / (difference of the rails' slopes dx/dy),
  with no focal length or pitch needed. Market St cable lines were standard gauge (4 ft 8.5 in) and strip centres
  are measured, so the effective gauge is 1435 + 60 mm (rail head). 1906 scan: **1.97 m** (IQR 1.91-2.01, 1715 of
  2231 frames, `data/rail_height.csv`); it also gives each frame's vanishing point (the horizon, after the grade).
- `scripts/lens_from_landmarks.py`: focal length from known sizes, three ways: the Ferry tower top (245 ft) against
  the rails' horizon and the map distance, the model's camera height against the rails, and the model's
  facade-to-facade width against Market St's 120 ft. On the 1906 scan they give 3237 / 3392 / 3409 px on the
  1618 px frame: **f = 3346 px (focal / width 2.07), hFOV 27.2 deg**, within 1% of the pilot bundle adjustment
  (27.0 deg). The first global run's 21.5 deg was wrong: `sfm_colmap.py` locked the lens only for the incremental
  mapper, and the global run drifted from its 3328 px start to 4215 px. Both mappers now hold it at
  `--focal-guess`. Re-run with the measured lens (`--focal-guess 2.068 --fixed-lens`, tag `global27`, 84 min of
  mapping): all 2231 frames, 0.79 px reprojection error, and the model now agrees with the known sizes it was not
  fitted to: camera height 1.99 m (rails 1.97), street 37.0 m (120 ft = 36.6). Re-running in a work folder whose
  database has another lens puts the new lens in and re-verifies the stored matches instead of matching again.
- Map placement: the ride does not start at 8th St. The Call Building (tallest south-side points, 3rd St) and the
  Ferry facade put the first camera (frame 1580) at ~800 m along the profile (near 5th St) and the last (10500) at
  ~2606 m, 140 m per global-model unit (in `sfm_global27` the fixes are at 4.37 and 13.055 units: first camera
  ~814 m, last ~2609 m). Use `--profile-range 814 2609` with `sfm_global27`, not `0 2487`: the road then matches
  the USGS profile to 0.7 m RMS (max 1.5 m), crest +5.5 m against the real +5.5 m (the old 21.5 deg model: +4.4 m).

- `scripts/metric_align.py`: metric, levelled copy of `sfm_global27/fixed_lens` -> `fixed_lens/metric/`. 139.9 m per
  model unit (the map fixes above); +X along the street = metres along `data/market_street_profile.csv`, +Z up =
  USGS elevation (pitch fitted so the cameras follow the profile), +Y left. Result: road 0.52 m RMS from the
  profile, crest +5.46 m (real +5.45), camera 2.0 m above the road in the cloud (rails 1.97).

## Align to a straight camera path

For a shot where the camera travels in a straight line, rotate the whole result so the path runs along +Z
and you look straight down it (level, first camera at the origin). Local, numpy only, works on either model's pulled output:

```bash
python scripts/align_to_path.py outputs/<movie>/lingbot-map      # -> .../lingbot-map/aligned/
```

It applies one rigid transform to points and cameras (stored as `align_matrix` in `poses.npz`). It does not straighten
wobble; it prints the RMS wobble and warns if the cameras aren't looking along the path. "Level" assumes the camera
was held roughly level on average. Add `--tumble` to also rotate 180 deg about the horizontal X axis (Y-up, path
running along -Z); that goes to `aligned_tumbled/` so the plain aligned result is kept.

## Compare two results side by side

Two Open3D windows, one per process, using Open3D's classic Visualizer (the one `view_ply.py` uses). Each process watches its own
camera and sends the 4x4 camera matrix to the other over a pipe, so orbit / pan / zoom in either window moves both; `+`/`-`
point size is synced, `R` resets both to a view from the first camera looking down the path, closing either closes both.
Each cloud is normalised (first camera at the origin, camera path length 1) because the models' scales differ.

```bash
python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled outputs/<movie>/lingbot-map/aligned_tumbled
```

Options: `--max-points` (default 1,000,000 per cloud), `--width/--height` per window, `--no-path` (hide the red camera path).
The sync protocol is tested with stand-in windows; real rendering is not. A native crash prints a Python traceback
(`faulthandler`); paste it if you hit one.

## LingBot-Map instead of LoGeR

```bash
modal run --detach modal_app.py::run --video ./clip.mp4 --model lingbot --last-seconds 10
modal run modal_app.py::pull --movie clip_last10s --variant lingbot-map
```

[LingBot-Map](https://github.com/robbyant/lingbot-map) is a streaming model. It also predicts **camera
intrinsics**, which `poses.npz` stores (`intrinsics`, `image_size`), so its output can feed a splat trainer.
Options: `--fps` (frames per second sampled, 0 = every frame; default 10), `--mode streaming|windowed`,
`--keyframe-interval` (0 = auto). It uses PyTorch attention rather than FlashInfer.

## Without Modal (plain Docker)

`docker/` has the same two images as Dockerfiles, plus an entrypoint that does what the Modal functions do
(fetch weights, optional `--last-seconds` trim, run, export) against a mounted `work/` folder with the same
layout as the volume. Needs an NVIDIA GPU and the NVIDIA Container Toolkit.

```bash
docker build -f docker/Dockerfile.loger   -t marketstreet-loger .
docker build -f docker/Dockerfile.lingbot -t marketstreet-lingbot .

mkdir -p work/clip/input && cp ./clip.mov work/clip/input/
docker run --rm --gpus all -v "$PWD/work:/work" marketstreet-loger   /work/clip/input/clip.mov --last-seconds 10
docker run --rm --gpus all -v "$PWD/work:/work" marketstreet-lingbot /work/clip/input/clip.mov --last-seconds 10
# results: work/clip_last10s/{LoGeR_star,lingbot-map}/{points.ply,poses.npz,trajectory.txt}
```

`--dry-run` prints the commands without running them; `--help` lists the options. The Dockerfiles, `docker/entrypoint.py`
and `modal_app.py` duplicate each other's steps, so keep them in sync (pinned commit SHAs, pip pins). Not built or run yet.

## Outputs on the volume (`<movie>` = filename stem)

| path | contents |
| --- | --- |
| `<movie>/<variant>/points.ply` | confidence-filtered, coloured point cloud |
| `<movie>/<variant>/poses.npz` | `poses` world_from_cam `[N,4,4]`, `frame_idx` source-frame indices |
| `<movie>/<variant>/trajectory.txt` | TUM format `ts tx ty tz qx qy qz qw` |
| `<movie>/<variant>/raw/*.pt` | raw LoGeR predictions; reload with upstream `demo_viser.py --load` |

## Caveats

- Not yet run end to end: the authoring sandbox cannot reach Hugging Face or Modal. Expect to iterate
  on the first image build / run.
- Upstream is pinned to commit `7685b7a` (`LOGER_SHA` in `modal_app.py`). Its README says it is a
  reimplementation; check the licence before using results commercially.
- LingBot-Map was smoke-tested on CPU with random weights only (plumbing, not quality). The checkpoint file
  name `lingbot-map.pt` and the pose convention (follows upstream's viewer) are unverified on real weights.
- GPU defaults to `A100-40GB`; use `LOGER_GPU=A100-80GB` if you hit OOM.
