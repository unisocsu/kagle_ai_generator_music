"""Build and push the persistent ACE-Step music worker to Kaggle."""
import base64
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

WORKER_SLUG = "ace-step-music-worker"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        print(output.strip(), flush=True)
    return result.returncode, output


def build_dir(user, job):
    root = pathlib.Path(__file__).resolve().parents[1]
    build = pathlib.Path("/tmp/kaggle-music-worker")
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)

    template = (root / "kernel" / "music_worker_bootstrap.py").read_text(encoding="utf-8")
    engine = (root / "kernel" / "music_worker.py").read_bytes()
    config_b64 = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    engine_b64 = base64.b64encode(engine).decode("ascii")
    template = template.replace("__MUSIC_WORKER_CONFIG_B64__", config_b64)
    template = template.replace("__MUSIC_WORKER_ENGINE_B64__", engine_b64)
    if "__MUSIC_WORKER_" in template:
        raise RuntimeError("Kaggle worker template still contains an unresolved placeholder")

    (build / "runner.py").write_text(template, encoding="utf-8")
    metadata = {
        "id": f"{user}/{job['kernel_slug']}",
        "title": "ACE-Step persistent music worker",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return build, metadata["id"]


def push(user, slug, job):
    job = dict(job, kernel_slug=slug)
    build, _ = build_dir(user, job)
    command = ["kaggle", "kernels", "push", "-p", str(build)]
    accelerator = os.environ.get("IN_ACCELERATOR", "NvidiaTeslaT4").strip()
    if accelerator:
        command += ["--accelerator", accelerator]
    print("Pushing private Kaggle worker:", slug, flush=True)
    return run(command)


def main():
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    gh_token = os.environ.get("GH_TOKEN", "").strip()
    owner = os.environ.get("GH_OWNER", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not all((user, gh_token, owner, repo)):
        raise SystemExit("Missing KAGGLE_USERNAME, GH_TOKEN, GH_OWNER, or GITHUB_REPOSITORY")

    try:
        idle_minutes = max(5, min(240, int(os.environ.get("IN_IDLE_MINUTES", "30"))))
    except ValueError:
        idle_minutes = 30

    job = {
        "repo": repo,
        "owner": owner,
        "gh_token": gh_token,
        "idle_minutes": idle_minutes,
        "max_hours": 5.0,
    }

    print("--- Current Kaggle worker status ---", flush=True)
    run(["kaggle", "kernels", "status", f"{user}/{WORKER_SLUG}"])
    print("------------------------------------", flush=True)

    code, output = push(user, WORKER_SLUG, job)
    if code != 0 and ("409" in output or "Conflict" in output):
        print("Kaggle reports an active kernel. Waiting 60 seconds before retry.", flush=True)
        time.sleep(60)
        code, output = push(user, WORKER_SLUG, job)

    if code != 0 and ("409" in output or "Conflict" in output):
        run_id = re.sub(r"[^0-9]", "", os.environ.get("GITHUB_RUN_ID", "")) or str(int(time.time()))
        fallback = f"{WORKER_SLUG}-{run_id[-10:]}"
        print("Existing kernel is still running; using a separate kernel id:", fallback, flush=True)
        code, output = push(user, fallback, job)

    if code != 0:
        raise SystemExit("Could not push the Kaggle music worker. Check Kaggle credentials, GPU access, and kernel status.")

    match = re.search(r"kaggle\.com/code/([^/\s]+)/([^/\s?#]+)", output)
    ref = f"{match.group(1)}/{match.group(2)}" if match else f"{user}/{WORKER_SLUG}"
    print("Kaggle worker ref:", ref, flush=True)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as stream:
            stream.write(f"ref={ref}\n")


if __name__ == "__main__":
    main()
