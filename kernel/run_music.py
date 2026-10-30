"""Kaggle entry point: load config, run the engine, and persist diagnostics."""
import base64
import json
import logging
import traceback
from pathlib import Path
from ace_step_engine import generate

HERE = Path(__file__).resolve().parent
OUTPUT = Path("/kaggle/working/music-output")
CONFIG_B64 = "__CONFIG_B64__"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(OUTPUT / "generation.log", encoding="utf-8")])
    logger = logging.getLogger("music-generator")
    try:
        config = json.loads(base64.b64decode(CONFIG_B64).decode("utf-8"))
        logger.info("Starting ACE-Step job; samples=%s duration=%s",
                    config["num_samples"], config["duration"])
        info = generate(config, OUTPUT)
        info.update({
            "prompt": config["prompt"], "duration": config["duration"],
            "num_samples_requested": config["num_samples"],
            "vocal_language": config.get("vocal_language", "en"),
            "instrumental": config.get("instrumental", False)})
        (OUTPUT / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Generation complete: %s", info["files"])
    except Exception:
        logger.exception("Generation failed")
        (OUTPUT / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
