"""Kaggle single-file bootstrap for the persistent ACE-Step worker.

scripts/push_music_worker.py embeds the private job configuration and
kernel/music_worker.py in this file before pushing it to Kaggle.
"""
import base64
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG_B64 = "__MUSIC_WORKER_CONFIG_B64__"
ENGINE_B64 = "__MUSIC_WORKER_ENGINE_B64__"

OUTPUT = Path("/kaggle/working/music-worker-output")
WORK = Path("/kaggle/temp/ace-step-music-worker") if Path("/kaggle/temp").exists() else Path("/tmp/ace-step-music-worker")
REPO = WORK / "ACE-Step-1.5"
PYTHON_VERSION = "3.12"


def emit(line):
    print(line, flush=True)


def run(cmd, cwd=None, env=None, check=True):
    cmd = [str(value) for value in cmd]
    emit("+ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        emit(line.rstrip())
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")
    return code


def main():
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(base64.b64decode(CONFIG_B64).decode("utf-8"))
        emit("ACE-Step persistent music worker starting")
        emit("Worker lifetime is bounded by Kaggle's session and quota limits.")
        run(["nvidia-smi"], check=False)

        WORK.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = [sys.executable, "-m", "uv"]

        if not REPO.exists():
            run(["git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", REPO])

        env = dict(
            os.environ,
            UV_CACHE_DIR=str(WORK / "uv-cache"),
            UV_LINK_MODE="copy",
            PYTHONUNBUFFERED="1",
            HF_HOME=str(WORK / "hf"),
            PYTHONPATH=str(REPO),
            ACESTEP_DISABLE_TQDM="1",
        )
        run(uv + ["sync", "--python", PYTHON_VERSION], cwd=REPO, env=env)

        engine = WORK / "music_worker.py"
        engine.write_bytes(base64.b64decode(ENGINE_B64))
        config_file = WORK / "worker-config.json"
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

        venv_python = REPO / ".venv" / "bin" / "python"
        run([venv_python, engine, config_file, OUTPUT, REPO], cwd=REPO, env=env)

        emit(f"Worker finished in {time.time() - started:.0f}s")
    except BaseException:
        failure = traceback.format_exc()
        emit(failure)
        (OUTPUT / "failure.txt").write_text(failure, encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
