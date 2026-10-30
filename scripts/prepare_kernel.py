"""Validates the workflow inputs and builds the Kaggle kernel folder (build/kernel).

Inputs arrive as environment variables (never interpolated into a shell command):
IN_PROMPT, IN_MODEL, IN_DURATION, IN_NUM_SAMPLES, IN_SEED, KAGGLE_USERNAME.
"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "kernel"
TEMPLATE = ROOT / "kernel" / "run_music.py"
PLACEHOLDER = "__CONFIG_B64__"

TITLE = "kagle ai music generator"
SLUG = "kagle-ai-music-generator"

MODELS = {
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
    "musicgen-large": "facebook/musicgen-large",
}


def fail(message):
    print(f"::error::{message}")
    sys.exit(1)


def read_number(name, default, low, high, cast):
    raw = os.environ.get(name, "").strip() or str(default)
    try:
        value = cast(raw)
    except ValueError:
        fail(f"{name} must be a number, got: {raw!r}")
    if not low <= value <= high:
        fail(f"{name} must be between {low} and {high}, got: {value}")
    return value


def main():
    owner = os.environ.get("KAGGLE_USERNAME", "").strip().lower()
    if not owner:
        fail("KAGGLE_USERNAME is empty (secret kagle_user_name is missing).")

    prompt = os.environ.get("IN_PROMPT", "").strip()
    if not prompt:
        fail("prompt is empty.")
    if len(prompt) > 500:
        fail("prompt is too long (max 500 characters).")

    model = os.environ.get("IN_MODEL", "").strip() or "musicgen-large"
    if model not in MODELS:
        fail(f"Unknown model {model!r}. Options: {', '.join(MODELS)}")

    config = {
        "prompt": prompt,
        "model": model,
        "model_id": MODELS[model],
        "duration": read_number("IN_DURATION", 15, 1, 30, float),
        "num_samples": read_number("IN_NUM_SAMPLES", 1, 1, 5, int),
        "seed": read_number("IN_SEED", -1, -1, 2**31 - 1, int),
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        fail(f"Template must contain {PLACEHOLDER} exactly once.")
    encoded = base64.b64encode(json.dumps(config, ensure_ascii=False).encode("utf-8")).decode("ascii")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / "run_music.py").write_text(template.replace(PLACEHOLDER, encoded), encoding="utf-8")

    metadata = {
        "id": f"{owner}/{SLUG}",
        "title": TITLE,
        "code_file": "run_music.py",
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
    (BUILD_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ROOT / "build" / "kernel_ref.txt").write_text(metadata["id"], encoding="utf-8")

    print(f"Kernel: {metadata['id']}")
    print(f"Model: {config['model_id']}, duration: {config['duration']}s, samples: {config['num_samples']}, seed: {config['seed']}")


if __name__ == "__main__":
    main()
