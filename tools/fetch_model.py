"""Idempotent GGUF downloader. CLI takes one or more model names from MODELS
(defaults to "qwen3-4b"). Each download is atomic — written to a `.tmp` sibling
and renamed on success.

Usage:
  uv run python -m tools.fetch_model                  # downloads "qwen3-4b"
  uv run python -m tools.fetch_model gemma3-27b       # downloads one good model
  uv run python -m tools.fetch_model qwen3-4b qwen3-32b
"""

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .common import MODELS_DIR

HF = "https://huggingface.co"


def _hf_token() -> str | None:
    """HF auth token from env or ~/.cache/huggingface/token (matches the
    huggingface-cli convention). Returns None if no token is configured."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if t := os.environ.get(var):
            return t.strip()
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.exists():
            return p.read_text().strip()
    return None

# name -> (url, filename)
MODELS: dict[str, tuple[str, str]] = {
    "qwen3-4b": (
        f"{HF}/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
        "Qwen3-4B-Q8_0.gguf",
    ),
    "gemma3-27b": (
        f"{HF}/google/gemma-3-27b-it-qat-q4_0-gguf/resolve/main/gemma-3-27b-it-q4_0.gguf",
        "gemma-3-27b-it-q4_0.gguf",
    ),
    "qwen3-30b-a3b": (
        f"{HF}/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q8_0.gguf",
        "Qwen3-30B-A3B-Q8_0.gguf",
    ),
    "qwen3-32b": (
        f"{HF}/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q8_0.gguf",
        "Qwen3-32B-Q8_0.gguf",
    ),
}

DEFAULT = "qwen3-4b"

CHUNK = 1 << 20  # 1 MiB
PROGRESS_EVERY = 64 << 20  # print every 64 MiB


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    headers = {"User-Agent": "honobono-fetch/1"}
    if token := _hf_token():
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp_cm = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(
                f"HTTP {e.code} on {url}\n"
                f"This model is gated. Accept its license at the HF page in a browser, "
                f"then set HF_TOKEN env var or run `huggingface-cli login` "
                f"(token currently {'set' if _hf_token() else 'not set'})."
            ) from e
        raise
    with resp_cm as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        next_mark = PROGRESS_EVERY
        with tmp.open("wb") as f:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if got >= next_mark:
                    pct = f" ({got / total * 100:.0f}%)" if total else ""
                    print(f"  {got >> 20} MiB / {total >> 20} MiB{pct}", flush=True)
                    next_mark += PROGRESS_EVERY
    if total and got != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"short read: got {got} of {total} bytes")
    tmp.rename(dest)


def fetch(name: str) -> Path:
    url, filename = MODELS[name]
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        print(f"{filename}: already present, skipping", flush=True)
        return dest
    print(f"downloading {url} -> {dest}", flush=True)
    download(url, dest)
    print(f"done: {dest} ({dest.stat().st_size >> 20} MiB)", flush=True)
    return dest


def main(argv: list[str]) -> int:
    names = argv[1:] or [DEFAULT]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}; known: {list(MODELS)}", file=sys.stderr)
        return 2
    for n in names:
        fetch(n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
