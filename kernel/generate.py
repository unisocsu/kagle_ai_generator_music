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
from contextlib import contextmanager
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
    # Tesla T4 is a pre-Ampere GPU. ACE-Step's normal CUDA dtype selection
    # chooses float16 on this hardware, which can overflow during XL-Turbo
    # diffusion and produce NaN latents. Override the actual loader dtype;
    # ACESTEP_DTYPE is not read by ACE-Step's CUDA initialization path.
    import torch

    handler = AceStepHandler()
    original_loader = handler._load_main_model_from_checkpoint

    def load_main_model_float32(**kwargs):
        handler.dtype = torch.float32
        log("[precision] Forcing DiT and dependent components to torch.float32")
        result = original_loader(**kwargs)
        for parameter in handler.model.parameters():
            if parameter.is_floating_point():
                if parameter.dtype != torch.float32:
                    raise RuntimeError(
                        f"Float32 precision override failed: first floating model parameter "
                        f"has dtype {parameter.dtype}"
                    )
                break
        else:
            raise RuntimeError("Float32 precision verification failed: model has no floating parameters")
        log("[precision] Verified DiT floating parameters are torch.float32")
        return result

    # initialize_service calls this instance method after choosing its
    # default dtype. Replacing the loader ensures the override happens
    # immediately before from_pretrained and remains active for VAE loading.
    handler._load_main_model_from_checkpoint = load_main_model_float32

    base = {
        "project_root": str(root),
        "config_path": DIT_NAME,
        "device": "cuda",
        "offload_to_cpu": True,
        "offload_dit_to_cpu": True,
        "quantization": None,
    }
    kwargs = accepted(handler.initialize_service, base, "initialize_service")
    log(f"== DiT init (forced float32, CPU offload, no quantization): {kwargs}")
    try:
        status, ok = handler.initialize_service(**kwargs)
        log(status)
        if not ok:
            raise RuntimeError(f"DiT initialization failed: {status}")
        if handler.dtype != torch.float32:
            raise RuntimeError(f"ACE-Step reverted dtype after initialization: {handler.dtype}")
        return handler
    except Exception:
        log(traceback.format_exc())
        del handler
        gc.collect()
        torch.cuda.empty_cache()
        raise



def install_audio_code_offload_patch(handler):
    """Decode LM audio codes without moving the full DiT to GPU.

    The normal _load_model_context("model") path transfers the entire 4B
    DiT to CUDA. Audio-code decoding only needs the small tokenizer
    quantizer and detokenizer, so move those two modules temporarily.
    """
    import types
    import torch

    def decode_audio_codes_cpu_offload(self, code_str):
        if self.model is None or not hasattr(self.model, "tokenizer") or not hasattr(self.model, "detokenizer"):
            return None

        code_ids = self._parse_audio_code_string(code_str)
        if not code_ids:
            return None

        quantizer = self.model.tokenizer.quantizer
        detokenizer = self.model.detokenizer
        device = torch.device(self.device)
        dtype = self.dtype

        log("[offload] Audio-code decode: moving only tokenizer.quantizer + detokenizer to CUDA")
        try:
            quantizer.to(device=device, dtype=dtype)
            detokenizer.to(device=device, dtype=dtype)
            with torch.inference_mode():
                indices = torch.tensor(code_ids, device=device, dtype=torch.long)
                indices = indices.unsqueeze(0).unsqueeze(-1)
                quantized = quantizer.get_output_from_indices(indices)
                if quantized.dtype != dtype:
                    quantized = quantized.to(dtype)
                result = detokenizer(quantized)
                log(f"[offload] Audio-code decode complete; latent shape={tuple(result.shape)}")
                return result
        finally:
            # Keep the large DiT and its audio-code components on CPU between
            # operations. The returned latent tensor remains on CUDA for the
            # subsequent generation step.
            quantizer.to("cpu")
            detokenizer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log("[offload] Tokenizer quantizer + detokenizer returned to CPU")

    handler._decode_audio_codes_to_latents = types.MethodType(decode_audio_codes_cpu_offload, handler)

    # The 4B XL DiT in FP32 cannot fit entirely in a single 15 GiB Tesla T4.
    # Keep its weights on CPU and stream the currently executing modules to
    # CUDA through Accelerate hooks. This is installed lazily on entry to the
    # first model context, after audio-code decoding has completed.
    original_load_context = handler._load_model_context
    offload_state = {"installed": False}

    @contextmanager
    def cpu_offload_model_context(self, model_name):
        if model_name != "model":
            with original_load_context(model_name):
                yield
            return

        if not offload_state["installed"]:
            from accelerate import cpu_offload

            gpu = torch.device(self.device)
            cuda_resident = [
                name for name, parameter in self.model.named_parameters()
                if parameter.device.type not in ("cpu", "meta")
            ]
            if cuda_resident:
                raise RuntimeError(
                    "Layerwise CPU offload expected the DiT on CPU before hook installation; "
                    f"found CUDA parameters, e.g. {cuda_resident[:5]}"
                )

            log("[offload] Installing Accelerate layerwise CPU offload for the full DiT")
            log("[offload] DiT weights stay in system RAM; active modules stream to GPU 0")
            self.model = cpu_offload(
                self.model,
                execution_device=gpu,
                offload_buffers=True,
                preload_module_classes=["AceStepEncoderLayer", "AceStepDiTLayer"],
            )
            meta_parameters = sum(
                1 for parameter in self.model.parameters()
                if parameter.device.type == "meta"
            )
            if meta_parameters == 0:
                raise RuntimeError(
                    "Accelerate CPU offload did not place DiT parameters on meta; "
                    "refusing to continue with a full-GPU model load."
                )
            offload_state["installed"] = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
                log(
                    f"[offload] Hooks installed for {meta_parameters} parameters; "
                    f"GPU free={free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB"
                )
            log("[offload] Full-model .to(cuda) is disabled; modules will stream layer-by-layer")

            # ACE-Step's turbo diffusion loop does not log inside its 8
            # decoder passes. Add lightweight timing hooks so a long CPU/GPU
            # transfer is distinguishable from a deadlock in future runs.
            if not offload_state.get("progress_hooks"):
                decoder = self.model.decoder
                original_decoder_forward = decoder.forward
                progress_state = {"pass": 0}

                def traced_decoder_forward(module, *args, **kwargs):
                    progress_state["pass"] += 1
                    pass_number = progress_state["pass"]
                    started = time.monotonic()
                    log(f"[progress] DiT pass {pass_number} started")
                    try:
                        result = original_decoder_forward(*args, **kwargs)
                        torch.cuda.synchronize(gpu)
                        log(
                            f"[progress] DiT pass {pass_number} completed "
                            f"in {time.monotonic() - started:.1f}s"
                        )
                        return result
                    except BaseException as exc:
                        log(
                            f"[progress] DiT pass {pass_number} failed after "
                            f"{time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}"
                        )
                        raise

                decoder.forward = types.MethodType(traced_decoder_forward, decoder)
                layers = list(decoder.layers)
                for layer_index, layer in enumerate(layers):
                    layer_state = {"started": None}

                    def layer_pre_hook(
                        module, inputs, index=layer_index, state=layer_state
                    ):
                        state["started"] = time.monotonic()
                        log(
                            f"[progress] Pass {progress_state['pass']} "
                            f"DiT layer {index + 1}/{len(layers)} started"
                        )

                    def layer_post_hook(
                        module, inputs, output, index=layer_index, state=layer_state
                    ):
                        started_at = state.get("started")
                        elapsed = time.monotonic() - started_at if started_at is not None else 0.0
                        log(
                            f"[progress] Pass {progress_state['pass']} "
                            f"DiT layer {index + 1}/{len(layers)} completed in {elapsed:.1f}s"
                        )

                    layer.register_forward_pre_hook(layer_pre_hook, prepend=True)
                    layer.register_forward_hook(layer_post_hook)

                offload_state["progress_hooks"] = True
                log(f"[progress] Installed timing hooks on {len(layers)} DiT layers")

        # Do not call the upstream _load_model_context("model"): it performs
        # model.to(cuda) on the entire 4B model and causes the reported OOM.
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            log("[offload] Layerwise DiT context finished; weights remain CPU-offloaded")

    handler._load_model_context = types.MethodType(cpu_offload_model_context, handler)
    log("[offload] Installed targeted audio-code + layerwise DiT CPU-offload patches")



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
    install_audio_code_offload_patch(dit)
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
