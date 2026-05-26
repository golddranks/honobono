import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from tools import common, fetch_episodes, fetch_model, pipeline

sys.stdout.reconfigure(line_buffering=True)

MODEL = "qwen3-4b"
EPS = [5]
RESUME_FROM: Path | None = Path("semantic_summary/2026-05-27_0346_Qwen3-4B-Q8_0")


def main() -> None:
    fetch_episodes.main()
    fetch_model.fetch(MODEL)

    _, filename = fetch_model.MODELS[MODEL]
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = common.SEMANTIC_SUMMARY_DIR / f"{run_id}_{Path(filename).stem}"
    print(f"MODEL={filename} EPS={EPS} OUT={out_dir} RESUME_FROM={RESUME_FROM}")

    registry: dict = {}
    if RESUME_FROM is not None:
        first_ep = min(EPS)
        prev_seq = first_ep - 1
        if prev_seq < 1:
            raise SystemExit(
                f"RESUME_FROM doesn't make sense with EPS starting at {first_ep} "
                "(nothing to resume from before ep1)"
            )
        snap = RESUME_FROM / "chars" / "10_register"
        matches = sorted(snap.glob(f"{prev_seq:03d}_*.json"))
        if not matches:
            raise SystemExit(f"no registry snapshot for ep{prev_seq} in {snap}")
        registry = json.loads(matches[0].read_text())
        src_summaries = RESUME_FROM / "summaries"
        dst_summaries = out_dir / "summaries"
        dst_summaries.mkdir(parents=True, exist_ok=True)
        for f in src_summaries.iterdir():
            # Filenames are zero-padded with the episode seq (e.g.
            # "004_n9629ex_ep004.json"). Only copy summaries for episodes
            # strictly before first_ep — anything later would correspond
            # to episodes we're about to re-process.
            if f.name[:3].isdigit() and int(f.name[:3]) >= first_ep:
                continue
            shutil.copy2(f, dst_summaries / f.name)

    for seq in EPS:
        pipeline.run_episode(seq, registry, model=filename, out_dir=out_dir)


if __name__ == "__main__":
    main()
