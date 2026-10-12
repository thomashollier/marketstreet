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
