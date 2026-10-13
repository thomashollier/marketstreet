"""LoGeR (long-context geometric reconstruction) on Modal.

One app, one shared Volume (`marketstreet-data`). A movie goes in, camera poses
and a coloured point cloud come out. Upstream: https://github.com/Junyi42/LoGeR

Run it from YOUR machine (needs `pip install modal`, `modal setup`, and the
movie file on local disk):

    modal run modal_app.py::run --video ./clip.mp4
    modal run modal_app.py::run --video ./clip.mp4 --variant LoGeR_star --stride 2
    modal run --detach modal_app.py::run --video ./clip.mp4      # survives closing the laptop

    modal run modal_app.py::pull --movie clip               # download results

    # same movie through LingBot-Map instead (https://github.com/robbyant/lingbot-map)
    modal run modal_app.py::run --video ./clip.mp4 --model lingbot --last-seconds 10
    modal run modal_app.py::pull --movie clip_last10s --variant lingbot-map

Layout on the volume (movie = video filename stem):

    /work/ckpts/<variant>/latest.pt        model weights, downloaded once
    /work/<movie>/input/<video>            uploaded movie
    /work/<movie>/<variant>/raw/*.pt       raw LoGeR predictions (points, conf, poses, images)
    /work/<movie>/<variant>/poses.npz      world_from_cam[N,4,4], frame_idx[N]
    /work/<movie>/<variant>/trajectory.txt TUM-style `ts tx ty tz qx qy qz qw`
    /work/<movie>/<variant>/points.ply     confidence-filtered coloured point cloud
    (variant = LoGeR | LoGeR_star | lingbot-map; lingbot's poses.npz also holds
     intrinsics[N,3,3] + image_size, and it has no raw/ dir)

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

LINGBOT_REPO = "https://github.com/robbyant/lingbot-map"
LINGBOT_SHA  = "6e6b0630c42a9505b891b881b21fd26a6cfff9f3"   # pinned; bump deliberately
LINGBOT_HF   = "robbyant/lingbot-map"
LINGBOT_CKPT = "lingbot-map.pt"      # file name inside the HF repo (README names it so)
LINGBOT_NAME = "lingbot-map"         # "variant" directory for its outputs
MODELS       = ["loger", "lingbot"]

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

# LingBot-Map needs torch 2.8.0 (cu128), LoGeR 2.6.0, so it gets its own image.
# Attention runs on PyTorch SDPA (upstream's --use_sdpa); FlashInfer would need
# nvcc for JIT kernels and is not installed.
lingbot_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    # upstream is tested with numpy<2; unpinned, lingbot-map's `opencv-python`
    # resolves to 5.x and drags numpy up to 2.2 (seen in the first build log).
    .pip_install("numpy<2", "opencv-python<4.12")
    .run_commands(
        f"git clone {LINGBOT_REPO} /opt/lingbot-map",
        f"git -C /opt/lingbot-map checkout {LINGBOT_SHA}",
        "pip install -e /opt/lingbot-map",
    )
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


def _out_movie(movie: str, last_seconds: float) -> str:
    """Volume namespace for results. A tail clip gets its own so runs with
    different --last-seconds don't overwrite each other."""
    return f"{movie}_last{last_seconds:g}s" if last_seconds > 0 else movie


def _trim_tail(src, dst, seconds: float):
    """Cut the last `seconds` of src into an H.264 mp4 at dst (ffmpeg)."""
    import subprocess
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-sseof", f"-{seconds:g}", "-i", str(src),
         "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)],
        check=True)


@app.function(image=loger_image, gpu=GPU, memory=32768,
              volumes={WORK: volume}, timeout=6 * 3600)
def reconstruct(movie: str, video_name: str, variant: str = "LoGeR_star",
                start_frame: int = 0, end_frame: int = -1, stride: int = 1,
                window_size: int = 32, overlap_size: int = 3,
                reset_every: int = 0, conf_percentile: float = 20.0,
                max_points: int = 2_000_000, last_seconds: float = 0.0) -> str:
    """Run upstream demo_viser.py headless (--skip_viser), then export poses + PLY.

    last_seconds > 0: first cut the last N seconds of the movie to an H.264 mp4
    (ffmpeg) and reconstruct that; results land under <movie>_last<N>s."""
    import subprocess

    volume.reload()   # pick up the upload / weights committed by other containers
    video = pathlib.Path(WORK, movie, "input", video_name)
    ckpt  = pathlib.Path(WORK, "ckpts", variant, "latest.pt")
    for p in (video, ckpt):
        if not p.exists():
            raise FileNotFoundError(f"{p} not on volume")

    out_movie = _out_movie(movie, last_seconds)
    if last_seconds > 0:
        tail = pathlib.Path(WORK, out_movie, "input", f"{out_movie}_{variant}.mp4")
        _trim_tail(video, tail, last_seconds)
        video = tail

    out = pathlib.Path(WORK, out_movie, variant)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seq = out_movie
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


@app.function(image=cpu_image, volumes={WORK: volume}, timeout=1800)
def fetch_lingbot_weights() -> str:
    """Download lingbot-map.pt from Hugging Face onto the volume (once)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError
    d = pathlib.Path(WORK, "ckpts", LINGBOT_NAME)
    dest = d / LINGBOT_CKPT
    if not dest.exists():
        try:
            hf_hub_download(LINGBOT_HF, LINGBOT_CKPT, local_dir=str(d))
        except EntryNotFoundError:
            raise SystemExit(f"{LINGBOT_CKPT} not in {LINGBOT_HF}; files there: "
                             f"{list_repo_files(LINGBOT_HF)}. Fix LINGBOT_CKPT.")
        volume.commit()
    print(f"[lingbot] weights: {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
    return str(dest)


@app.function(image=lingbot_image, gpu=GPU, memory=32768,
              volumes={WORK: volume}, timeout=6 * 3600)
def reconstruct_lingbot(movie: str, video_name: str, fps: float = 10.0,
                        mode: str = "streaming", keyframe_interval: int = 0,
                        conf_percentile: float = 20.0, max_points: int = 2_000_000,
                        last_seconds: float = 0.0) -> str:
    """Run scripts/run_lingbot.py (headless LingBot-Map) on the movie."""
    import subprocess

    volume.reload()
    video = pathlib.Path(WORK, movie, "input", video_name)
    ckpt = pathlib.Path(WORK, "ckpts", LINGBOT_NAME, LINGBOT_CKPT)
    for p in (video, ckpt):
        if not p.exists():
            raise FileNotFoundError(f"{p} not on volume")

    out_movie = _out_movie(movie, last_seconds)
    if last_seconds > 0:
        tail = pathlib.Path(WORK, out_movie, "input", f"{out_movie}_{LINGBOT_NAME}.mp4")
        _trim_tail(video, tail, last_seconds)
        video = tail

    out = pathlib.Path(WORK, out_movie, LINGBOT_NAME)
    subprocess.run(
        ["python", "/app/scripts/run_lingbot.py",
         "--video", str(video), "--model-path", str(ckpt), "--out", str(out),
         "--fps", str(fps), "--mode", mode, "--keyframe-interval", str(keyframe_interval),
         "--conf-percentile", str(conf_percentile), "--max-points", str(max_points)],
        check=True, cwd="/opt/lingbot-map")
    volume.commit()
    return str(out)


@app.function(image=loger_image, cpu=8.0, memory=98304, volumes={WORK: volume}, timeout=7200)
def fuse_best(movie: str, variant: str = "LoGeR_star", start_frame: int = 0, end_frame: int = -1,
              stride: int = 1, extra: list = None) -> str:
    """Best-observation fusion of a finished LoGeR run's raw predictions (CPU only, no model).
    Writes <movie>/<variant>/fused/{points.ply,poses.npz,fusion_stats.json}. See scripts/fuse_best.py."""
    import subprocess
    volume.reload()
    out = pathlib.Path(WORK, movie, variant)
    pred = out / "raw" / f"{movie}_{start_frame}_{end_frame}_{stride}.pt"
    if not pred.exists():
        raise FileNotFoundError(f"{pred} not on volume; run the full reconstruct first")
    subprocess.run(["python", "/app/scripts/fuse_best.py", "--pred", str(pred),
                    "--out", str(out / "fused"), *(extra or [])], check=True)
    volume.commit()
    return str(out / "fused")


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
         last_seconds: float = 0.0,
         model: str = "loger",
         fps: float = 10.0,
         mode: str = "streaming",
         keyframe_interval: int = 0,
         skip_upload: bool = False):
    """Upload the movie to the volume, fetch weights if needed, reconstruct.

    --model             loger (default) or lingbot (LingBot-Map). With lingbot, results
                        are under <movie>/lingbot-map and the options below apply
                        instead of --variant/--stride/--window-size/--overlap-size/
                        --reset-every/--start-frame/--end-frame.
    --fps               (lingbot) frames sampled per second of video, 0 = every frame
    --mode              (lingbot) streaming, or windowed for very long clips
    --keyframe-interval (lingbot) 0 = auto (bounds the KV cache beyond 320 frames)

    --last-seconds      use only the last N seconds of the movie (0 = whole movie).
                        Results go under <stem>_last<N>s; --start/end-frame and
                        --stride then count within that tail clip.

    --variant           LoGeR or LoGeR_star (upstream's demo runs both; _star is the SE(3) variant)
    --start/end-frame   source-frame range, end -1 = last
    --stride            keep every Nth source frame
    --window-size       frames per LoGeR window; --overlap-size frames shared between windows
    --reset-every       reset TTT state every N windows; use for >1k-frame clips
    --conf-percentile   drop the lowest N% confidence points from points.ply
    --skip-upload       the movie is already on the volume at <stem>/input/<name>
                        (e.g. via `modal volume put marketstreet-data ./clip.mp4 /clip/input/clip.mp4`)
    """
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; valid: {MODELS}")
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; valid: {VARIANTS}")
    if last_seconds < 0:
        raise SystemExit("--last-seconds must be >= 0")
    src = pathlib.Path(video)
    movie = src.stem

    if not skip_upload:
        if not src.is_file():
            raise SystemExit(f"{src} not found locally")
        print(f"[loger] uploading {src} ({src.stat().st_size / 1e6:.1f} MB) "
              f"-> {VOLUME_NAME}:/{movie}/input/{src.name}", flush=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(src), f"/{movie}/input/{src.name}")

    if model == "lingbot":
        variant = LINGBOT_NAME
        fetch_lingbot_weights.remote()
        out = reconstruct_lingbot.remote(movie, src.name, fps, mode, keyframe_interval,
                                         conf_percentile, max_points, last_seconds)
    else:
        fetch_weights.remote(variant)
        out = reconstruct.remote(movie, src.name, variant, start_frame, end_frame, stride,
                                 window_size, overlap_size, reset_every,
                                 conf_percentile, max_points, last_seconds)
    print(f"[loger] done: {VOLUME_NAME}:{out}", flush=True)
    print(f"[loger] download: modal run modal_app.py::pull --movie {_out_movie(movie, last_seconds)} "
          f"--variant {variant}",
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
def fuse(movie: str, variant: str = "LoGeR_star", start_frame: int = 0, end_frame: int = -1,
         stride: int = 1, voxel: float = 0.0, rel_err: float = 0.03, tier: float = 0.5,
         min_frames: int = 2, pixel_stride: int = 1, conf_pct: float = 20.0, far_pct: float = 98.0,
         rtol: float = 0.03, min_cos: float = 0.01, max_points: int = 4_000_000):
    """Keep the best observation of each surface from a finished LoGeR run (no GPU, no re-run of the model).
    Use the same --start-frame/--end-frame/--stride as the original run. It prints what each stage removed.
    Too few points? --min-frames 1 (also keeps moving objects), --voxel (smaller = denser but stricter on support),
    --conf-pct 10, --rtol 0.1 (fewer flying-pixel drops). Too noisy? --min-frames 3, --rel-err 0.05.
    More in scripts/fuse_best.py."""
    extra = ["--voxel", voxel, "--rel-err", rel_err, "--tier", tier, "--min-frames", min_frames,
             "--pixel-stride", pixel_stride, "--conf-pct", conf_pct, "--far-pct", far_pct,
             "--rtol", rtol, "--min-cos", min_cos, "--max-points", max_points]
    out = fuse_best.remote(movie, variant, start_frame, end_frame, stride, [str(x) for x in extra])
    print(f"[loger] done: {VOLUME_NAME}:{out}", flush=True)
    print(f"[loger] download: modal run modal_app.py::pull --movie {movie} --variant {variant}/fused", flush=True)


@app.local_entrypoint()
def pull(movie: str, variant: str = "LoGeR_star", download: str = "./outputs",
         raw: bool = False):
    """Download poses.npz, trajectory.txt, points.ply (and fusion_stats.json / raw/ if present).
    For fused results pass --variant LoGeR_star/fused."""
    import subprocess
    dest = pathlib.Path(download) / movie / variant
    dest.mkdir(parents=True, exist_ok=True)
    items = ["poses.npz", "trajectory.txt", "points.ply", "fusion_stats.json"] + (["raw"] if raw else [])
    for name in items:
        r = subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME,
                            f"/{movie}/{variant}/{name}", str(dest)])
        print(f"[loger] {dest / name}" if r.returncode == 0 else f"[loger] {name}: not on the volume (skipped)")
