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

One Open3D window, two viewports, one shared camera, so orbit / pan / zoom in either half moves both identically.
Each cloud is normalised (first camera at the origin, camera path length 1) because the models' scales differ.

```bash
python scripts/compare_view.py outputs/<movie>/LoGeR_star/aligned_tumbled outputs/<movie>/lingbot-map/aligned_tumbled
```

Left-drag orbit, right-drag or ctrl+left-drag pan, wheel zoom; `R` reset, `P` toggle the red camera path, `+`/`-` point size.
`--max-points` (default 1,000,000 per cloud) subsamples if it is slow. The camera logic is tested; the window itself is not.

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
