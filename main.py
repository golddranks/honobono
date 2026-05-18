from datetime import datetime
from pathlib import Path

from tools import common, fetch_episodes, fetch_model, pipeline

MODEL = "gemma3-27b"
EPS = [1, 2, 3, 4, 5]


def main() -> None:
    fetch_episodes.main()
    fetch_model.fetch(MODEL)

    _, filename = fetch_model.MODELS[MODEL]
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = common.SEMANTIC_SUMMARY_DIR / f"{run_id}_{Path(filename).stem}"
    print(f"MODEL={filename} EPS={EPS} OUT={out_dir}", flush=True)

    registry: dict = {}
    for seq in EPS:
        pipeline.run_episode(seq, registry, model=filename, out_dir=out_dir)


if __name__ == "__main__":
    main()
