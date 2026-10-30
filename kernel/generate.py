"""ACE-Step engine.

Runs INSIDE ACE-Step's own uv environment (Python 3.12 + the torch build the
project pins), launched by kernel/run_music.py. It never touches Kaggle's
system Python packages.

Usage: python generate.py <config.json> <output_dir> <repo_root>
"""
import gc
import inspect
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

DIT_NAME = "acestep-v15-xl-turbo"
LM_NAME = "acestep-5Hz-lm-1.7B"


def log(*args):
    print(*args, flush=True)


def accepted(target, kwargs, label):
    """Keep only the keyword arguments this ACE-Step version understands."""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        log(f"[compat] {label}: ignoring unsupported arguments {dropped}")
    return kept


def download_models(root):
    ckpt = root / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    log("== Downloading base checkpoints (VAE, text encoder, default models)")
    try:
        subprocess.run([sys.executable, "-m", "acestep.model_downloader"], cwd=root, check=True)
    except Exception as exc:  # fall back to a direct download below
        log(f"[warn] acestep.model_downloader failed: {exc}")
    from huggingface_hub import snapshot_download
    if not (ckpt / "vae").exists():
        log("== Fetching shared components directly from Hugging Face")
        snapshot_download(repo_id="ACE-Step/Ace-Step1.5", local_dir=str(ckpt),
                          allow_patterns=["vae/*", "Qwen3-Embedding-0.6B/*"])
    for name in (DIT_NAME, LM_NAME):
        target = ckpt / name
        if not target.exists() or not any(target.iterdir()):
            log(f"== Downloading {name}")
            snapshot_download(repo_id=f"ACE-Step/{name}", local_dir=str(target))
    return ckpt


def init_dit(root, AceStepHandler):
    attempts = [
        {"offload_to_cpu": True, "offload_dit_to_cpu": True, "quantization": "int8_weight_only"},
        {"offload_to_cpu": True, "offload_dit_to_cpu": True, "quantization": None},
    ]
    last = "no attempt made"
    for number, extra in enumerate(attempts, 1):
        handler = AceStepHandler()
        base = {"project_root": str(root), "config_path": DIT_NAME, "device": "cuda"}
        try:
            kwargs = accepted(handler.initialize_service, {**base, **extra}, "initialize_service")
            log(f"== DiT init attempt {number}: {kwargs}")
            status, ok = handler.initialize_service(**kwargs)
            log(status)
            if ok:
                return handler
            last = str(status)
        except Exception:
            last = traceback.format_exc()
            log(last)
        del handler
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    raise RuntimeError(f"DiT initialization failed: {last}")


def init_lm(ckpt, LLMHandler):
    handler = LLMHandler()
    base = {"checkpoint_dir": str(ckpt), "lm_model_path": LM_NAME,
            "backend": "pt", "device": "cuda", "offload_to_cpu": True}
    try:
        status, ok = handler.initialize(**accepted(handler.initialize, base, "LLMHandler.initialize"))
        log(status)
        if ok:
            return handler, True
    except Exception:
        log(traceback.format_exc())
    log("[warn] language model unavailable; continuing without LM planning (thinking=False)")
    return handler, False


def main():
    config_path, output_dir, root = (Path(a) for a in sys.argv[1:4])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Check that the Kaggle kernel was given a GPU accelerator.")
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(f"{torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GiB)")
        log(f"GPU {i}: {gpus[-1]}")

    ckpt = download_models(root)

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit = init_dit(root, AceStepHandler)
    llm, llm_ok = init_lm(ckpt, LLMHandler)

    lyrics = "[Instrumental]" if config["instrumental"] and not config["lyrics"] else config["lyrics"]
    files, seeds = [], []
    for index in range(config["num_samples"]):
        seed = config["seed"] + index if config["seed"] >= 0 else -1
        params = GenerationParams(**accepted(GenerationParams, {
            "task_type": "text2music",
            "caption": config["prompt"],
            "lyrics": lyrics,
            "instrumental": config["instrumental"],
            "vocal_language": config["vocal_language"],
            "duration": float(config["duration"]),
            "inference_steps": 8,
            "shift": 3.0,
            "seed": seed,
            "thinking": llm_ok,
            "use_cot_caption": llm_ok,
            "use_cot_metas": llm_ok,
        }, "GenerationParams"))
        gen_config = GenerationConfig(**accepted(GenerationConfig, {
            "batch_size": 1,
            "audio_format": "wav",
            "use_random_seed": seed < 0,
            "seeds": [seed] if seed >= 0 else None,
        }, "GenerationConfig"))
        log(f"== Generating sample {index + 1}/{config['num_samples']} (seed={seed})")
        result = generate_music(dit, llm, params, gen_config, save_dir=str(output_dir))
        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")
        for audio in result.audios:
            source = Path(audio["path"])
            target = output_dir / f"music_{len(files) + 1:02d}.wav"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            files.append(target.name)
            seeds.append(audio.get("params", {}).get("seed", seed))

    info = {
        "engine": "ACE-Step 1.5",
        "dit_model": DIT_NAME,
        "lm_model": LM_NAME if llm_ok else None,
        "gpus": gpus,
        "prompt": config["prompt"],
        "lyrics_chars": len(config["lyrics"]),
        "duration": config["duration"],
        "vocal_language": config["vocal_language"],
        "instrumental": config["instrumental"],
        "seeds": seeds,
        "files": files,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log("== Done:", files)


if __name__ == "__main__":
    main()
