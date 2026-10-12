"""Run one movie through the model baked into this image, outside Modal.

Mirrors fetch_weights / reconstruct / reconstruct_lingbot in modal_app.py
(keep them in sync) against a mounted /work folder that uses the same layout
as the `marketstreet-data` volume:

    /work/<movie>/input/<video>        your movie (movie = filename stem)
    /work/ckpts/...                    weights, downloaded on first run
    /work/<movie>/<variant>/...        results (poses.npz, points.ply, ...)

    docker run --rm --gpus all -v "$PWD/work:/work" marketstreet-loger \\
        /work/clip/input/clip.mov --last-seconds 10

MODEL (loger | lingbot) is set by the image. --dry-run prints the commands.
"""
import argparse
import os
import pathlib
import subprocess

MODEL = os.environ["MODEL"]
HF_LOGER, HF_LINGBOT, LINGBOT_CKPT = "Junyi42/LoGeR", "robbyant/lingbot-map", "lingbot-map.pt"
DRY = False


def run(cmd, **kw):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if not DRY:
        subprocess.run(cmd, check=True, **kw)


def main():
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="movie path inside the container, e.g. /work/clip/input/clip.mov")
    ap.add_argument("--work", default="/work")
    ap.add_argument("--last-seconds", type=float, default=0.0)
    ap.add_argument("--conf-percentile", type=float, default=20.0)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_argument_group("loger only")
    g.add_argument("--variant", default="LoGeR_star", choices=["LoGeR", "LoGeR_star"])
    g.add_argument("--start-frame", type=int, default=0)
    g.add_argument("--end-frame", type=int, default=-1)
    g.add_argument("--stride", type=int, default=1)
    g.add_argument("--window-size", type=int, default=32)
    g.add_argument("--overlap-size", type=int, default=3)
    g.add_argument("--reset-every", type=int, default=0)
    g = ap.add_argument_group("lingbot only")
    g.add_argument("--fps", type=float, default=10.0, help="0 = every frame")
    g.add_argument("--mode", default="streaming", choices=["streaming", "windowed"])
    g.add_argument("--keyframe-interval", type=int, default=0, help="0 = auto")
    a = ap.parse_args()
    DRY = a.dry_run

    work = pathlib.Path(a.work)
    video = pathlib.Path(a.video)
    movie = video.stem
    variant = a.variant if MODEL == "loger" else "lingbot-map"
    out_movie = f"{movie}_last{a.last_seconds:g}s" if a.last_seconds > 0 else movie

    # weights (once)
    if MODEL == "loger":
        ckpt = work / "ckpts" / variant / "latest.pt"
        repo, fname, local = HF_LOGER, f"{variant}/latest.pt", work / "ckpts"
    else:
        ckpt = work / "ckpts" / "lingbot-map" / LINGBOT_CKPT
        repo, fname, local = HF_LINGBOT, LINGBOT_CKPT, work / "ckpts" / "lingbot-map"
    if not ckpt.exists():
        run(["python", "-c", f"from huggingface_hub import hf_hub_download as d; "
             f"d({repo!r}, {fname!r}, local_dir={str(local)!r})"])

    # optional tail trim, same ffmpeg call as modal_app._trim_tail
    if a.last_seconds > 0:
        tail = work / out_movie / "input" / f"{out_movie}_{variant}.mp4"
        tail.parent.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-sseof", f"-{a.last_seconds:g}", "-i", video,
             "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", tail])
        video = tail

    out = work / out_movie / variant
    if MODEL == "loger":
        raw = out / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        cmd = ["python", "demo_viser.py", "--input", video,
               "--config", f"ckpts/{variant}/original_config.yaml", "--model_name", ckpt,
               "--start_frame", a.start_frame, "--end_frame", a.end_frame, "--stride", a.stride,
               "--window_size", a.window_size, "--overlap_size", a.overlap_size,
               "--skip_viser", "--seq_name", out_movie, "--output_folder", raw,
               "--output_txt", out / "trajectory.txt"]
        if a.reset_every > 0:
            cmd += ["--reset_every", a.reset_every]
        run(cmd, cwd="/opt/loger")
        run(["python", "/app/scripts/export_loger.py",
             "--pred", raw / f"{out_movie}_{a.start_frame}_{a.end_frame}_{a.stride}.pt",
             "--out", out, "--start-frame", a.start_frame, "--stride", a.stride,
             "--conf-percentile", a.conf_percentile, "--max-points", a.max_points])
    else:
        run(["python", "/app/scripts/run_lingbot.py", "--video", video, "--model-path", ckpt,
             "--out", out, "--fps", a.fps, "--mode", a.mode,
             "--keyframe-interval", a.keyframe_interval,
             "--conf-percentile", a.conf_percentile, "--max-points", a.max_points],
            cwd="/opt/lingbot-map")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
