"""Persistent ACE-Step music worker for Kaggle.

Loads the smaller 2B Turbo DiT and 0.6B planner once, then processes music
requests queued as GitHub Issues until idle timeout, stop request, or time limit.
"""
import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import requests

DIT_NAME = "acestep-v15-turbo"
LM_NAME = "acestep-5Hz-lm-0.6B"
JOB_MARKER = "<!--ace-music-job:v1-->"
EVENT_MARKER = "<!--ace-music-event:v1-->"
RESULT_MARKER = "<!--ace-music-result:v1-->"
STATUS_MARKER = "<!--ace-music-worker-status:v1-->"
STATUS_TITLE = "[music-worker-status]"
JOB_PREFIX = "[music-job] "


def log(*args):
    print(*args, flush=True)


def accepted(target, values):
    import inspect
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return values
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return values
    return {k: v for k, v in values.items() if k in params}


def download_models(root):
    from huggingface_hub import snapshot_download

    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    log("== Preparing only the selected persistent-worker checkpoints")

    # Avoid the default model_downloader because it also fetches the 1.7B
    # planner. This worker deliberately uses the smaller 0.6B planner.
    snapshot_download(
        repo_id="ACE-Step/Ace-Step1.5",
        local_dir=str(checkpoints),
        allow_patterns=["vae/**", "Qwen3-Embedding-0.6B/**"],
    )

    for name in (DIT_NAME, LM_NAME):
        target = checkpoints / name
        if not target.exists() or not any(target.iterdir()):
            log("[download] Fetching selected component:", name)
            snapshot_download(
                repo_id=f"ACE-Step/{name}",
                local_dir=str(target),
            )

    return checkpoints


def init_models(root):
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle worker has no CUDA GPU. Enable the GPU accelerator.")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        log(
            f"[hardware] GPU {index}: {torch.cuda.get_device_name(index)} "
            f"({props.total_memory / 1024**3:.1f} GiB)"
        )

    checkpoints = download_models(root)

    dit = AceStepHandler()
    dit_kwargs = accepted(dit.initialize_service, {
        "project_root": str(root),
        "config_path": DIT_NAME,
        "device": "cuda",
        # Keep the 2B DiT resident on the GPU across songs. The VAE and text
        # encoder still use ACE-Step's normal CPU offload contexts.
        "offload_to_cpu": True,
        "offload_dit_to_cpu": False,
        "quantization": None,
        "use_mlx_dit": False,
    })
    log("== Loading persistent DiT:", DIT_NAME)
    status, ok = dit.initialize_service(**dit_kwargs)
    log(status)
    if not ok:
        raise RuntimeError(f"DiT initialization failed: {status}")

    llm = LLMHandler()
    log("== Loading persistent planner:", LM_NAME)
    status, ok = llm.initialize(**accepted(llm.initialize, {
        "checkpoint_dir": str(checkpoints),
        "lm_model_path": LM_NAME,
        "backend": "pt",
        "device": "cuda",
        "offload_to_cpu": False,
    }))
    log(status)
    if not ok:
        log("[warn] Planner unavailable; continuing with ACE-Step's no-LM mode")
        llm = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        log(f"[hardware] GPU 0 free after model load: {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB")

    return dit, llm


class GitHub:
    def __init__(self, repo, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
            raise ValueError("Invalid GitHub repository name")
        self.repo = repo
        self.api = "https://api.github.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ACE-Step-Music-Worker/1.0",
        })

    def call(self, method, path, body=None, params=None):
        response = self.session.request(
            method,
            f"{self.api}/repos/{self.repo}{path}",
            json=body,
            params=params,
            timeout=45,
        )
        try:
            data = response.json() if response.content else None
        except ValueError:
            data = None
        if response.status_code >= 400:
            message = (data or {}).get("message", response.text[:300] if response.text else "")
            raise RuntimeError(f"GitHub API {method} {path}: HTTP {response.status_code}: {message}")
        return data

    def comments(self, issue_number):
        result = []
        page = 1
        while True:
            batch = self.call(
                "GET", f"/issues/{issue_number}/comments",
                params={"per_page": 100, "page": page},
            ) or []
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1

    def comment(self, issue_number, text):
        return self.call("POST", f"/issues/{issue_number}/comments", {"body": text})

    def event(self, issue_number, state, message, extra=None):
        data = {"state": state, "message": message, "ts": int(time.time())}
        if extra:
            data.update(extra)
        self.comment(issue_number, EVENT_MARKER + "\n" + json.dumps(data, ensure_ascii=False))


def find_status_issue(gh):
    issues = gh.call("GET", "/issues", params={
        "state": "open", "per_page": 100, "sort": "created", "direction": "desc",
    }) or []
    for issue in issues:
        if "pull_request" not in issue and issue.get("title") == STATUS_TITLE:
            return issue["number"]
    return None


def ensure_status_issue(gh):
    number = find_status_issue(gh)
    if number:
        return number
    created = gh.call("POST", "/issues", {
        "title": STATUS_TITLE,
        "body": "ACE-Step music worker is starting.",
    })
    return created["number"]


def heartbeat(gh, number, state, details=None):
    payload = {"state": state, "ts": int(time.time()), "model": DIT_NAME, "planner": LM_NAME}
    if details:
        payload.update(details)
    gh.call("PATCH", f"/issues/{number}", {
        "body": STATUS_MARKER + "\n" + json.dumps(payload, ensure_ascii=False)
    })


def read_worker_state(gh, number):
    issue = gh.call("GET", f"/issues/{number}")
    body = issue.get("body") or ""
    if not body.startswith(STATUS_MARKER):
        return ""
    try:
        return json.loads(body[len(STATUS_MARKER):].strip()).get("state", "")
    except ValueError:
        return ""


def parse_job_body(body):
    start = (body or "").find(JOB_MARKER)
    if start < 0:
        return None
    raw = (body or "")[start + len(JOB_MARKER):].strip()
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    prompt = str(data.get("prompt", "")).strip()
    if not prompt or len(prompt) > 512:
        return None
    try:
        duration = float(data.get("duration", 30))
        seed = int(data.get("seed", -1))
    except (TypeError, ValueError):
        return None
    if not 10 <= duration <= 180 or not (-1 <= seed <= 2**31 - 1):
        return None
    lyrics = str(data.get("lyrics", ""))[:4096]
    instrumental = bool(data.get("instrumental", False))
    language = str(data.get("vocal_language", "en"))
    if language not in {"en", "he", "es", "fr", "de", "ja", "zh", "unknown"}:
        language = "en"
    return {
        "prompt": prompt,
        "lyrics": lyrics,
        "instrumental": instrumental,
        "vocal_language": language,
        "duration": duration,
        "seed": seed,
    }


def create_release(gh, issue_number, prompt, output_files):
    tag = f"ace-music-{issue_number}-{int(time.time())}"
    release = gh.call("POST", "/releases", {
        "tag_name": tag,
        "name": f"ACE-Step track for job #{issue_number}",
        "body": "Generated by the persistent ACE-Step Kaggle worker.\n\nPrompt: " + prompt[:500],
        "draft": False,
        "prerelease": False,
    })
    upload_url = (release.get("upload_url") or "").split("{", 1)[0]
    if not upload_url:
        raise RuntimeError("GitHub did not return a release upload URL")

    assets = []
    for index, path in enumerate(output_files, 1):
        file_path = Path(path)
        asset_name = f"music-{issue_number}-{index:02d}.wav"
        with file_path.open("rb") as stream:
            response = gh.session.post(
                upload_url,
                params={"name": asset_name},
                data=stream,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300,
            )
        try:
            item = response.json()
        except ValueError:
            item = {}
        if response.status_code >= 400:
            raise RuntimeError(
                f"Release asset upload failed: HTTP {response.status_code}: "
                + str(item.get("message", response.text[:300]))
            )
        assets.append({
            "name": item.get("name", asset_name),
            "size": item.get("size", file_path.stat().st_size),
            "url": item.get("browser_download_url"),
        })
    return {
        "tag": tag,
        "release_url": release.get("html_url"),
        "assets": assets,
    }


def generate_track(dit, llm, job, issue_number, output_dir):
    import torch
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    output_dir.mkdir(parents=True, exist_ok=True)
    lyrics = job["lyrics"]
    if job["instrumental"] and not lyrics.strip():
        lyrics = "[Instrumental]"

    params = GenerationParams(**accepted(GenerationParams, {
        "task_type": "text2music",
        "caption": job["prompt"],
        "lyrics": lyrics,
        "instrumental": job["instrumental"],
        "vocal_language": job["vocal_language"],
        "duration": float(job["duration"]),
        "inference_steps": 8,
        "shift": 3.0,
        "seed": int(job["seed"]),
        "thinking": llm is not None,
        "use_cot_caption": llm is not None,
        "use_cot_metas": llm is not None,
    }))
    seed = int(job["seed"])
    config = GenerationConfig(**accepted(GenerationConfig, {
        "batch_size": 1,
        "audio_format": "wav",
        "use_random_seed": seed < 0,
        "seeds": [seed] if seed >= 0 else None,
    }))

    log(f"[job #{issue_number}] Generating {job['duration']}s track")
    result = generate_music(dit, llm, params, config, save_dir=str(output_dir))
    if not result.success:
        raise RuntimeError(f"ACE-Step generation failed: {result.error}")

    files = []
    for audio in result.audios:
        source = Path(audio["path"])
        if source.exists() and source.suffix.lower() == ".wav":
            files.append(source)
    if not files:
        raise RuntimeError("ACE-Step returned success but no WAV file was found")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return files, result


def process_issue(gh, issue, dit, llm, status_number):
    issue_number = issue["number"]
    job = parse_job_body(issue.get("body", ""))
    if not job:
        gh.event(issue_number, "failed", "בקשת היצירה אינה תקינה. בדוק את ההגדרות ונסה שוב.")
        gh.call("PATCH", f"/issues/{issue_number}", {"state": "closed"})
        return

    gh.event(issue_number, "running", "הבקשה התקבלה. המודל כבר טעון בזיכרון ומתחיל ליצור.")
    heartbeat(gh, status_number, "processing", {"job": issue_number})
    output_dir = Path(tempfile.mkdtemp(prefix=f"ace-music-{issue_number}-", dir="/kaggle/working"))

    try:
        files, result = generate_track(dit, llm, job, issue_number, output_dir)
        gh.event(issue_number, "uploading", "היצירה הסתיימה. מעלה את קובץ האודיו.")
        release = create_release(gh, issue_number, job["prompt"], files)
        payload = {
            "state": "complete",
            "message": "היצירה מוכנה.",
            "release_url": release["release_url"],
            "assets": release["assets"],
            "duration": job["duration"],
            "prompt": job["prompt"],
        }
        gh.comment(issue_number, RESULT_MARKER + "\n" + json.dumps(payload, ensure_ascii=False))
        gh.call("PATCH", f"/issues/{issue_number}", {"state": "closed"})
        log(f"[job #{issue_number}] Complete: {release['release_url']}")
    except Exception as exc:
        log(f"[job #{issue_number}] ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        try:
            gh.comment(issue_number, RESULT_MARKER + "\n" + json.dumps({
                "state": "failed",
                "message": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }, ensure_ascii=False))
            gh.call("PATCH", f"/issues/{issue_number}", {"state": "closed"})
        except Exception:
            traceback.print_exc()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def find_pending_jobs(gh, owner):
    issues = gh.call("GET", "/issues", params={
        "state": "open", "per_page": 100, "sort": "created", "direction": "asc",
    }) or []
    pending = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        if not str(issue.get("title", "")).startswith(JOB_PREFIX):
            continue
        if (issue.get("user") or {}).get("login", "").lower() != owner.lower():
            continue
        if not parse_job_body(issue.get("body", "")):
            continue
        pending.append(issue)
    return pending


def worker_loop(gh, owner, dit, llm, config, status_number):
    idle_seconds = max(5, float(config.get("idle_minutes", 30)) * 60)
    max_seconds = max(60, float(config.get("max_hours", 5.0)) * 3600)
    poll_seconds = 8
    started = last_activity = last_heartbeat = time.time()
    handled = set()
    reason = "unknown"
    count = 0

    heartbeat(gh, status_number, "online")
    log(f"[worker] Ready. Idle timeout={idle_seconds / 60:.0f} min; max runtime={max_seconds / 3600:.1f} h")

    while True:
        now = time.time()
        if now - started >= max_seconds:
            reason = "max-runtime"
            break
        if now - last_activity >= idle_seconds:
            reason = "idle-timeout"
            break

        if now - last_heartbeat >= 30:
            last_heartbeat = now
            state = read_worker_state(gh, status_number)
            if state == "stop":
                reason = "stop-requested"
                break
            heartbeat(gh, status_number, "online", {"completed_jobs": count})

        try:
            pending = find_pending_jobs(gh, owner)
            for issue in pending:
                if issue["number"] in handled:
                    continue
                handled.add(issue["number"])
                last_activity = time.time()
                log(f"[worker] Processing job #{issue['number']}")
                process_issue(gh, issue, dit, llm, status_number)
                count += 1
                last_activity = time.time()
                heartbeat(gh, status_number, "online", {"completed_jobs": count})
            time.sleep(poll_seconds)
        except Exception as exc:
            log("[worker] Polling error:", type(exc).__name__, exc)
            traceback.print_exc()
            time.sleep(15)

    heartbeat(gh, status_number, "offline", {"reason": reason, "completed_jobs": count})
    log(f"[worker] Stopped: {reason}; completed jobs={count}")
    return reason, count


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: music_worker.py <config.json> <output_dir> <ace_step_root>")
    config_path, output_dir, root = (Path(arg) for arg in sys.argv[1:4])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = config.get("repo", "")
    token = config.get("gh_token", "")
    owner = config.get("owner", "")
    if not repo or not token or not owner:
        raise RuntimeError("Worker configuration is missing repo, GitHub token, or owner")

    gh = GitHub(repo, token)
    status_number = ensure_status_issue(gh)
    heartbeat(gh, status_number, "starting")
    dit = llm = None
    try:
        dit, llm = init_models(root)
        worker_loop(gh, owner, dit, llm, config, status_number)
    except BaseException as exc:
        log("[worker] Fatal error:", type(exc).__name__, exc)
        traceback.print_exc()
        try:
            heartbeat(gh, status_number, "offline", {"error": str(exc)[:1000]})
        except Exception:
            pass
        raise
    finally:
        try:
            if llm is not None:
                llm.unload()
        except Exception:
            pass
        del llm, dit
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    (output_dir / "worker-result.json").write_text(
        json.dumps({"state": "stopped", "ts": int(time.time())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
