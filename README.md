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
`--end-frame`, `--last-seconds N` (only the last N seconds; results go under `<stem>_last<N>s`), `--window-size`, `--overlap-size`, `--reset-every` (clips >1k frames).
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

Use the same `--start-frame/--end-frame/--stride` as the original run. `fusion_stats.json` lists the settings and per-frame candidate
counts. Validated on a synthetic street with distance-growing, per-frame depth errors, outliers and a moving object (median error
0.027 vs 0.072 for a naive merge, floaters 0% vs 3%); defaults are not yet tuned on real footage.

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
