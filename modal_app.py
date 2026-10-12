"""LoGeR (long-context geometric reconstruction) on Modal.

One app, one shared Volume (`marketstreet-data`). A movie goes in, camera poses
and a coloured point cloud come out. Upstream: https://github.com/Junyi42/LoGeR

Run it from YOUR machine (needs `pip install modal`, `modal setup`, and the
movie file on local disk):

    modal run modal_app.py::run --video ./clip.mp4
    modal run modal_app.py::run --video ./clip.mp4 --variant LoGeR_star --stride 2
    modal run --detach modal_app.py::run --video ./clip.mp4      # survives closing the laptop

    modal run modal_app.py::pull --movie clip               # download results

Layout on the volume (movie = video filename stem):

    /work/ckpts/<variant>/latest.pt        model weights, downloaded once
    /work/<movie>/input/<video>            uploaded movie
    /work/<movie>/<variant>/raw/*.pt       raw LoGeR predictions (points, conf, poses, images)
    /work/<movie>/<variant>/poses.npz      world_from_cam[N,4,4], frame_idx[N]
    /work/<movie>/<variant>/trajectory.txt TUM-style `ts tx ty tz qx qy qz qw`
    /work/<movie>/<variant>/points.ply     confidence-filtered coloured point cloud

The raw .pt can be re-opened interactively on any machine with the LoGeR repo:
    python demo_viser.py --load <raw dir> --input <video> --model_name x --config x

GPU: A100-40GB by default (LoGeR streams the video in windows, so memory does
not grow with clip length). Override at deploy time: LOGER_GPU=A100-80GB modal run ...
"""
import os
import pathlib

import modal

APP_NAME    = "marketstreet-loger"
VOLUME_NAME = "marketstreet-data"
WORK        = "/work"

LOGER_REPO  = "https://github.com/Junyi42/LoGeR"
LOGER_SHA   = "7685b7a41949e62ed8a69309642465aee4cb89ba"   # pinned; bump deliberately
HF_REPO     = "Junyi42/LoGeR"
VARIANTS    = ["LoGeR", "LoGeR_star"]

GPU = os.environ.get("LOGER_GPU", "A100-40GB")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --------------------------------- images -----------------------------------
# Upstream is not pip-installable: clone it and run from /opt/loger so that
# `import loger` and the relative ckpts/<variant>/original_config.yaml resolve.
# add_local_* must come last in the chain (Modal rule).
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)

loger_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone {LOGER_REPO} /opt/loger",
        f"git -C /opt/loger checkout {LOGER_SHA}",
        # upstream's own pins: torch==2.6.0, numpy==1.26.4, viser, evo, ...
        "pip install -r /opt/loger/requirements.txt",
    )
    # imported by loger/ but missing from upstream's requirements.txt
    .pip_install("omegaconf", "onnxruntime", "requests", "tqdm")
    .add_local_dir("scripts", "/app/scripts")
)

app = modal.App(APP_NAME)


# =================================  STAGES  =================================
@app.function(image=cpu_image, volumes={WORK: volume}, timeout=1800)
def fetch_weights(variant: str) -> str:
    """Download <variant>/latest.pt from Hugging Face onto the volume (once)."""
    from huggingface_hub import hf_hub_download
    dest = pathlib.Path(WORK, "ckpts", variant, "latest.pt")
    if not dest.exists():
        hf_hub_download(HF_REPO, f"{variant}/latest.pt", local_dir=f"{WORK}/ckpts")
        volume.commit()
    print(f"[loger] weights: {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
    return str(dest)


@app.function(image=loger_image, gpu=GPU, memory=32768,
              volumes={WORK: volume}, timeout=6 * 3600)
def reconstruct(movie: str, video_name: str, variant: str = "LoGeR_star",
                start_frame: int = 0, end_frame: int = -1, stride: int = 1,
                window_size: int = 32, overlap_size: int = 3,
                reset_every: int = 0, conf_percentile: float = 20.0,
                max_points: int = 2_000_000) -> str:
    """Run upstream demo_viser.py headless (--skip_viser), then export poses + PLY."""
    import subprocess

    volume.reload()   # pick up the upload / weights committed by other containers
    video = pathlib.Path(WORK, movie, "input", video_name)
    ckpt  = pathlib.Path(WORK, "ckpts", variant, "latest.pt")
    for p in (video, ckpt):
        if not p.exists():
            raise FileNotFoundError(f"{p} not on volume")

    out = pathlib.Path(WORK, movie, variant)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seq = movie
    # demo_viser.py names its output f"{seq_name}_{start}_{end}_{stride}.pt"
    pred = raw / f"{seq}_{start_frame}_{end_frame}_{stride}.pt"

    cmd = [
        "python", "demo_viser.py",
        "--input", str(video),
        "--config", f"ckpts/{variant}/original_config.yaml",
        "--model_name", str(ckpt),
        "--start_frame", str(start_frame), "--end_frame", str(end_frame),
        "--stride", str(stride),
        "--window_size", str(window_size), "--overlap_size", str(overlap_size),
        "--skip_viser",
        "--seq_name", seq,
        "--output_folder", str(raw),
        "--output_txt", str(out / "trajectory.txt"),
    ]
    if reset_every > 0:   # upstream: use for extreme-long sequences (>1k frames)
        cmd += ["--reset_every", str(reset_every)]
    print("[loger] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/opt/loger")
    volume.commit()

    if not pred.exists():
        raise FileNotFoundError(f"expected predictions at {pred}; got {sorted(raw.iterdir())}")
    _export(pred, out, start_frame, stride, conf_percentile, max_points)
    return str(out)


def _export(pred, out, start_frame, stride, conf_percentile, max_points):
    import subprocess
    subprocess.run(
        ["python", "/app/scripts/export_loger.py",
         "--pred", str(pred), "--out", str(out),
         "--start-frame", str(start_frame), "--stride", str(stride),
         "--conf-percentile", str(conf_percentile), "--max-points", str(max_points)],
        check=True)
    volume.commit()


@app.function(image=loger_image, memory=32768, volumes={WORK: volume}, timeout=1800)
def export_only(movie: str, variant: str = "LoGeR_star", start_frame: int = 0,
                end_frame: int = -1, stride: int = 1, conf_percentile: float = 20.0,
                max_points: int = 2_000_000) -> str:
    """Re-run just the poses/PLY export from saved raw predictions (CPU, no inference)."""
    volume.reload()
    out = pathlib.Path(WORK, movie, variant)
    pred = out / "raw" / f"{movie}_{start_frame}_{end_frame}_{stride}.pt"
    if not pred.exists():
        raise FileNotFoundError(f"{pred} not on volume; run the full reconstruct first")
    _export(pred, out, start_frame, stride, conf_percentile, max_points)
    return str(out)


# ============================  LOCAL ENTRYPOINTS  ===========================
@app.local_entrypoint()
def run(video: str,
         variant: str = "LoGeR_star",
         start_frame: int = 0,
         end_frame: int = -1,
         stride: int = 1,
         window_size: int = 32,
         overlap_size: int = 3,
         reset_every: int = 0,
         conf_percentile: float = 20.0,
         max_points: int = 2_000_000,
         skip_upload: bool = False):
    """Upload the movie to the volume, fetch weights if needed, reconstruct.

    --variant           LoGeR or LoGeR_star (upstream's demo runs both; _star is the SE(3) variant)
    --start/end-frame   source-frame range, end -1 = last
    --stride            keep every Nth source frame
    --window-size       frames per LoGeR window; --overlap-size frames shared between windows
    --reset-every       reset TTT state every N windows; use for >1k-frame clips
    --conf-percentile   drop the lowest N% confidence points from points.ply
    --skip-upload       the movie is already on the volume at <stem>/input/<name>
                        (e.g. via `modal volume put marketstreet-data ./clip.mp4 /clip/input/clip.mp4`)
    """
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; valid: {VARIANTS}")
    src = pathlib.Path(video)
    movie = src.stem

    if not skip_upload:
        if not src.is_file():
            raise SystemExit(f"{src} not found locally")
        print(f"[loger] uploading {src} ({src.stat().st_size / 1e6:.1f} MB) "
              f"-> {VOLUME_NAME}:/{movie}/input/{src.name}", flush=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(src), f"/{movie}/input/{src.name}")

    fetch_weights.remote(variant)
    out = reconstruct.remote(movie, src.name, variant, start_frame, end_frame, stride,
                             window_size, overlap_size, reset_every,
                             conf_percentile, max_points)
    print(f"[loger] done: {VOLUME_NAME}:{out}", flush=True)
    print(f"[loger] download: modal run modal_app.py::pull --movie {movie} --variant {variant}",
          flush=True)


@app.local_entrypoint()
def export(movie: str, variant: str = "LoGeR_star", start_frame: int = 0,
           end_frame: int = -1, stride: int = 1, conf_percentile: float = 20.0,
           max_points: int = 2_000_000):
    """Re-export poses.npz + points.ply from saved raw predictions, no GPU.
    Use the same --start-frame/--end-frame/--stride as the original run.
    Handy for changing --conf-percentile / --max-points."""
    out = export_only.remote(movie, variant, start_frame, end_frame, stride,
                             conf_percentile, max_points)
    print(f"[loger] done: {VOLUME_NAME}:{out}", flush=True)


@app.local_entrypoint()
def pull(movie: str, variant: str = "LoGeR_star", download: str = "./outputs",
         raw: bool = False):
    """Download poses.npz, trajectory.txt and points.ply (and raw/ with --raw)."""
    import subprocess
    dest = pathlib.Path(download) / movie / variant
    dest.mkdir(parents=True, exist_ok=True)
    items = ["poses.npz", "trajectory.txt", "points.ply"] + (["raw"] if raw else [])
    for name in items:
        subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME,
                        f"/{movie}/{variant}/{name}", str(dest)], check=True)
        print(f"[loger] {dest / name}")
