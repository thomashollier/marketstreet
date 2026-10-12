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
`--end-frame`, `--window-size`, `--overlap-size`, `--reset-every` (clips >1k frames).
For a very large movie, pre-upload with
`modal volume put marketstreet-data ./clip.mp4 /clip/input/clip.mp4` and pass `--skip-upload`.

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
- GPU defaults to `A100-40GB`; use `LOGER_GPU=A100-80GB` if you hit OOM.
