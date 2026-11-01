"""Kaggle entry point. Standard library only.

Kaggle runs exactly one script per kernel, so scripts/prepare_kernel.py embeds
the validated run config and kernel/generate.py into this file's two constants
below. At runtime this script
  1. installs `uv`,
  2. clones ACE-Step and builds its own environment with `uv sync` (Python 3.12),
  3. runs generate.py inside that environment,
  4. leaves only small results in /kaggle/working/music-output.

Heavy files (repo, venv, model weights) live in a scratch directory so that
`kaggle kernels output` never has to download them.
"""
import base64
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG_B64 = "__CONFIG_B64__"
ENGINE_B64 = "__ENGINE_B64__"

OUTPUT = Path("/kaggle/working/music-output")
WORK = Path("/kaggle/temp/ace-step") if Path("/kaggle/temp").exists() else Path("/tmp/ace-step")
REPO = WORK / "ACE-Step-1.5"
PYTHON_VERSION = "3.12"

_log_file = None


def emit(line):
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def run(cmd, cwd=None, env=None, check=True):
    cmd = [str(c) for c in cmd]
    emit(f"+ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        emit(line.rstrip())
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")
    return code


def main():
    global _log_file
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT / "generation.log", "a", encoding="utf-8")
    started = time.time()
    try:
        config = json.loads(base64.b64decode(CONFIG_B64).decode("utf-8"))
        emit(f"Job: samples={config['num_samples']} duration={config['duration']}s "
             f"lang={config['vocal_language']} instrumental={config['instrumental']}")
        WORK.mkdir(parents=True, exist_ok=True)

        run(["nvidia-smi"], check=False)
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = [sys.executable, "-m", "uv"]

        if not REPO.exists():
            run(["git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", REPO])
        # Precision is enforced inside kernel/generate.py at the actual
        # ACE-Step model loader. ACESTEP_DTYPE is not honored by its CUDA path.
        env = dict(os.environ, UV_CACHE_DIR=str(WORK / "uv-cache"), UV_LINK_MODE="copy",
                   PYTHONUNBUFFERED="1", HF_HOME=str(WORK / "hf"), PYTHONPATH=str(REPO))
        emit("ACE-Step precision override: runtime DiT loader will force torch.float32 (Tesla T4)")
        run(uv + ["sync", "--python", PYTHON_VERSION], cwd=REPO, env=env)

        engine = WORK / "generate.py"
        engine.write_bytes(base64.b64decode(ENGINE_B64))
        config_file = WORK / "config.json"
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

        venv_python = REPO / ".venv" / "bin" / "python"
        run([venv_python, engine, config_file, OUTPUT, REPO], cwd=REPO, env=env)

        if not list(OUTPUT.glob("*.wav")):
            raise RuntimeError("Engine finished but produced no WAV files.")
        emit(f"Finished in {time.time() - started:.0f}s")
    except BaseException:
        text = traceback.format_exc()
        emit(text)
        (OUTPUT / "failure.txt").write_text(text, encoding="utf-8")
        _log_file.close()
        sys.exit(1)
    _log_file.close()


if __name__ == "__main__":
    main()
