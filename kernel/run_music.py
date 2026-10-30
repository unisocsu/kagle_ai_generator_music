"""Runs inside a Kaggle kernel: generates music with MusicGen and writes WAV files to /kaggle/working.

The placeholder below is replaced by scripts/prepare_kernel.py with the run settings
(base64-encoded JSON), so the prompt never touches a shell.
"""
import base64
import json
import os
import subprocess
import sys
import time

CONFIG = json.loads(base64.b64decode("__CONFIG_B64__").decode("utf-8"))
OUT_DIR = "/kaggle/working"


def install_deps():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "scipy"],
        check=True,
    )


def main():
    install_deps()

    import numpy as np
    import torch
    from scipy.io import wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    os.makedirs(OUT_DIR, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    model_id = CONFIG["model_id"]
    print(f"device={device} gpu={gpu_name} model={model_id}", flush=True)
    if not use_cuda:
        print("WARNING: no GPU found, generation will be very slow.", flush=True)

    started = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)

    sampling_rate = model.config.audio_encoder.sampling_rate
    frame_rate = model.config.audio_encoder.frame_rate
    max_new_tokens = int(CONFIG["duration"] * frame_rate)
    inputs = processor(text=[CONFIG["prompt"]], padding=True, return_tensors="pt").to(device)

    files = []
    for i in range(CONFIG["num_samples"]):
        if CONFIG["seed"] >= 0:
            torch.manual_seed(CONFIG["seed"] + i)
        t0 = time.time()
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens,
            )
        wav = audio_values[0, 0].float().cpu().numpy()
        peak = float(np.abs(wav).max())
        if peak > 1.0:
            wav = wav / peak
        name = f"music_{i + 1:02d}.wav"
        wavfile.write(os.path.join(OUT_DIR, name), sampling_rate, wav.astype(np.float32))
        files.append(name)
        print(f"saved {name} in {time.time() - t0:.0f}s", flush=True)

    info = {
        "config": CONFIG,
        "gpu": gpu_name,
        "sampling_rate": sampling_rate,
        "files": files,
        "total_seconds": round(time.time() - started),
    }
    with open(os.path.join(OUT_DIR, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
