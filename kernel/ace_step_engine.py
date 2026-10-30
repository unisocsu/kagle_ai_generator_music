"""ACE-Step model lifecycle and generation. Kept separate from Kaggle orchestration."""
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def generate(config, output_dir):
    root = Path("/kaggle/working/ACE-Step-1.5")
    checkpoints = root / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(root)])
    # Kaggle images may not include the PEP 517 backend required by ACE-Step.
    # Keep setuptools compatible with the Kaggle image PyTorch build (requires <82).
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "hatchling", "setuptools<82"])
    _run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"], cwd=root)
    _run([sys.executable, "-m", "acestep.model_downloader"], cwd=root)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="ACE-Step/acestep-v15-xl-turbo", local_dir=str(checkpoints / "acestep-v15-xl-turbo"))

    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    gpu_names = []
    print("CUDA devices:", torch.cuda.device_count(), flush=True)
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        gpu_names.append(name)
        print(f"GPU {index}: {name} / {props.total_memory / 1024**3:.1f} GiB", flush=True)

    dit, llm = AceStepHandler(), LLMHandler()
    status, ok = dit.initialize_service(
        project_root=str(root), config_path="acestep-v15-xl-turbo",
        device="cuda", offload_to_cpu=True, offload_dit_to_cpu=True,
        quantization="int8_weight_only", use_mlx_dit=False)
    if not ok:
        raise RuntimeError(f"DiT initialization failed: {status}")
    print(status, flush=True)

    status, ok = llm.initialize(
        checkpoint_dir=str(checkpoints), lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt", device="cuda", offload_to_cpu=True)
    if not ok:
        raise RuntimeError(f"LM initialization failed: {status}")
    print(status, flush=True)

    files = []
    for index in range(config["num_samples"]):
        seed = config["seed"] + index if config["seed"] >= 0 else -1
        params = GenerationParams(
            task_type="text2music", caption=config["prompt"],
            duration=float(config["duration"]),
            vocal_language=config.get("vocal_language", "en"),
            instrumental=config.get("instrumental", False),
            lyrics=config.get("lyrics", ""), seed=seed,
            inference_steps=8, shift=3.0, thinking=True,
            use_cot_caption=True, use_cot_metas=True)
        result = generate_music(
            dit, llm, params,
            GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=(seed < 0)),
            save_dir=str(output_dir))
        if not result.success:
            raise RuntimeError(f"ACE-Step generation failed: {result.error}")
        for audio in result.audios:
            source = Path(audio["path"])
            destination = output_dir / f"music_{index + 1:02d}.wav"
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            files.append(destination.name)

    return {
        "engine": "ACE-Step 1.5",
        "dit_model": "acestep-v15-xl-turbo (4B)",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "files": files,
    }
