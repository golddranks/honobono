"""Harness utilities: paths, episode IO, JSON IO, in-process llama.cpp
client, the `Result` carrier, and the on-disk Result writer.

Pipeline data logic (registry mutations, candidate pruning, prior-summary
loading) lives in core.py.
"""

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

# --- paths -----------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
EPISODES_DIR = REPO_DIR / "episodes"
MODELS_DIR = REPO_DIR / "models"
SEMANTIC_SUMMARY_DIR = REPO_DIR / "semantic_summary"


# --- episode IO -----------------------------------------------------------
def episode_stem(seq: int) -> str:
    """File stem (e.g. '001_n9629ex_ep001') for a given episode seq."""
    return next(EPISODES_DIR.glob(f"{seq:03d}_*.txt")).stem


def read_episode(seq: int) -> str:
    """Episode text including the title as line 1, blank line, then body."""
    path = next(EPISODES_DIR.glob(f"{seq:03d}_*.txt"))
    return path.read_text(encoding="utf-8").strip()


# --- JSON IO --------------------------------------------------------------
def dump_json(obj: Any, path: Path | str) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path | str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# --- Result carrier -------------------------------------------------------


@dataclass
class Result:
    """Carrier for one model invocation. `output` is the raw text from
    generate() (or whatever the core stage transforms it into)."""

    prompt: str
    meta: dict
    output: Any


# --- llama.cpp client -----------------------------------------------------

_llm: Any = None
_llm_path: Path | None = None
_llm_n_ctx: int | None = None


def _resolve_model(model: str) -> Path:
    p = Path(model)
    return p if p.is_absolute() else MODELS_DIR / model


def _get_llm(model: str, num_ctx: int):
    """Lazy singleton. Reloads if the requested model path or n_ctx changes."""
    global _llm, _llm_path, _llm_n_ctx
    from llama_cpp import Llama

    path = _resolve_model(model)
    if _llm is not None and _llm_path == path and _llm_n_ctx == num_ctx:
        return _llm
    _llm = None  # drop reference so the old model can be freed
    print(f"    [load] {path.name} n_ctx={num_ctx}", flush=True)
    _llm = Llama(
        model_path=str(path),
        n_ctx=num_ctx,
        n_gpu_layers=-1,
        verbose=False,
    )
    _llm_path = path
    _llm_n_ctx = num_ctx
    return _llm


def generate(
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    num_ctx: int = 32768,
    num_predict: int = 2000,
    fmt: dict | Literal["json"] | None = None,
) -> Result:
    """Run one in-process llama.cpp completion. Returns a Result with the
    raw model text in `output`. `fmt` is a JSON Schema dict (constrains the
    output via grammar) or "json" (any JSON object). No retries — failures
    raise."""
    llm = _get_llm(model, num_ctx)

    grammar = None
    if fmt is not None:
        from llama_cpp import LlamaGrammar

        schema = fmt if isinstance(fmt, dict) else {"type": "object"}
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)

    t0 = time.perf_counter_ns()
    resp = llm(
        prompt,
        max_tokens=num_predict,
        temperature=temperature,
        grammar=grammar,
    )
    elapsed_ns = time.perf_counter_ns() - t0

    choice = resp["choices"][0]
    text = choice["text"].strip()
    if not text:
        raise RuntimeError(f"empty response: {resp}")
    done = choice["finish_reason"]
    usage = resp["usage"]
    pin = usage["prompt_tokens"]
    eout = usage["completion_tokens"]
    pct = pin / num_ctx * 100
    warn = " !!INPUT-NEAR-CTX" if pin >= num_ctx * 0.95 else ""
    if done == "length":
        warn += " !!OUTPUT-TRUNCATED"
    print(
        f"    [gen] prompt={pin}/{num_ctx} ({pct:.0f}%) "
        f"eval={eout}/{num_predict} done={done}{warn}",
        flush=True,
    )
    meta = {
        "model": str(_llm_path),
        "done_reason": done,
        "total_duration_ns": elapsed_ns,
        "prompt_eval_count": pin,
        "eval_count": eout,
    }
    return Result(prompt=prompt, meta=meta, output=text)


# --- Result on-disk writer ------------------------------------------------


_META_KEYS = (
    "model",
    "done_reason",
    "total_duration_ns",
    "prompt_eval_count",
    "eval_count",
)


def dump_meta(payload, path) -> None:
    """Save the small useful subset of a generation meta dict."""
    dump_json({k: payload.get(k) for k in _META_KEYS}, path)


def save_result(out_dir: Path, stem: str, result: Result) -> None:
    """Standardized 3-file layout per Result:
    <out_dir>/<stem>.json         result.output
    <out_dir>/<stem>.prompt.txt   result.prompt
    <out_dir>/<stem>.meta.json    filtered meta
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = result.output
    if is_dataclass(output) and not isinstance(output, type):
        output = asdict(output)
    dump_json(output, out_dir / f"{stem}.json")
    (out_dir / f"{stem}.prompt.txt").write_text(result.prompt, encoding="utf-8")
    dump_meta(result.meta, out_dir / f"{stem}.meta.json")
