"""Wait for a Kaggle worker kernel to stop and record its final status."""
import os
import subprocess
import sys
import time


def status(ref):
    result = subprocess.run(
        ["kaggle", "kernels", "status", ref],
        capture_output=True,
        text=True,
    )
    text = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    return result.returncode, text


def main():
    ref = os.environ.get("KERNEL_REF", "").strip()
    if not ref:
        raise SystemExit("KERNEL_REF is empty")

    max_wait = int(os.environ.get("MAX_WAIT_MINUTES", "330")) * 60
    deadline = time.time() + max_wait
    last = ""
    while time.time() < deadline:
        code, output = status(ref)
        if output and output != last:
            print(time.strftime("%H:%M:%S"), output, flush=True)
            last = output
        lower = output.lower()
        if any(word in lower for word in ("complete", "error", "cancelled", "canceled", "failed")):
            break
        if code != 0 and any(word in lower for word in ("not found", "does not exist")):
            raise SystemExit(output)
        time.sleep(45)
    else:
        print("Wait limit reached; Kaggle worker may still be running.", flush=True)
        with open("music-worker-result.txt", "w", encoding="utf-8") as stream:
            stream.write("timeout\n")
        return

    with open("music-worker-result.txt", "w", encoding="utf-8") as stream:
        stream.write(output + "\n")
    print("Kaggle music worker stopped.", flush=True)


if __name__ == "__main__":
    main()
