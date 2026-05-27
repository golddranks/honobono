import sys
import traceback
from datetime import datetime
from pathlib import Path

from tools import common, fetch_episodes, fetch_model, pipeline

sys.stdout.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]

# Each model is run from scratch over `EPS`. If a model crashes mid-run,
# the loop moves on to the next model. Use Ctrl-C to stop the batch.
MODELS_TO_RUN = ["gemma4-31b"]
EPS = list(range(1, 17))


def main() -> None:
    fetch_episodes.main()
    for key in MODELS_TO_RUN:
        try:
            fetch_model.fetch(key)
        except (Exception, SystemExit):
            print(f"!! fetch failed for {key}; skipping")
            traceback.print_exc()

    for key in MODELS_TO_RUN:
        _, filename = fetch_model.MODELS[key]
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = common.SEMANTIC_SUMMARY_DIR / f"{run_id}_{Path(filename).stem}"
        print(f"\n=== model: {key} ({filename}) ===")
        print(f"MODEL={filename} EPS={EPS} OUT={out_dir}")
        registry: dict = {}
        try:
            for seq in EPS:
                pipeline.run_episode(seq, registry, model=filename, out_dir=out_dir)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit):
            print(f"!! pipeline crashed on {key}; moving on to next model")
            traceback.print_exc()


if __name__ == "__main__":
    main()
