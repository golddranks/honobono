import json
import shutil
from datetime import datetime
from pathlib import Path

from tools import common, fetch_episodes, fetch_model, pipeline

MODEL = "qwen3-4b"
EPS = [3, 4, 5]
RESUME_FROM: Path | None = Path("semantic_summary/2026-05-27_0207_Qwen3-4B-Q8_0")


def main() -> None:
    fetch_episodes.main()
    fetch_model.fetch(MODEL)

    _, filename = fetch_model.MODELS[MODEL]
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = common.SEMANTIC_SUMMARY_DIR / f"{run_id}_{Path(filename).stem}"
    print(f"MODEL={filename} EPS={EPS} OUT={out_dir} RESUME_FROM={RESUME_FROM}", flush=True)

    registry: dict = {}
    if RESUME_FROM is not None:
        registry = json.loads((RESUME_FROM / "chars" / "registry.json").read_text())
        src_summaries = RESUME_FROM / "summaries"
        dst_summaries = out_dir / "summaries"
        dst_summaries.mkdir(parents=True, exist_ok=True)
        for f in src_summaries.iterdir():
            shutil.copy2(f, dst_summaries / f.name)

    for seq in EPS:
        pipeline.run_episode(seq, registry, model=filename, out_dir=out_dir)


if __name__ == "__main__":
    main()
