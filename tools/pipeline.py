"""Per-episode pipeline stage. Driver (main.py) constructs the registry,
picks the model + out_dir, and loops over episodes.

Per ep, top to bottom: extract → verify → (ep1 short-circuit OR merge_eval →
merge_judge → merge_prune → update_reg_b → consolidate → update_history →
update_reg_c) → summarize. The registry mutates in place across eps; file
IO happens at clearly visible save_result / dump_json sites only.

Output layout (under out_dir):
  chars/
    01_extract/<stem>.{json,prompt.txt,meta.json}
    02_extract_verify/<stem>.{json,prompt.txt,meta.json}
    03_merge_eval/<stem>/<cname>.{json,prompt.txt,meta.json}   (ep2+)
    04_merge_judge/<stem>/<cname>.{json,prompt.txt,meta.json}  (ep2+)
    05_merge_prune/<stem>/<cname>.{json,prompt.txt,meta.json}  (ep2+)
    06_consolidate/<stem>/<cid>.{json,prompt.txt,meta.json}    (ep2+)
    07_update_history/<stem>/<cid>.{json,prompt.txt,meta.json} (ep2+)
    registry.json
  summaries/<stem>.{json,prompt.txt,meta.json}
"""

import time
from pathlib import Path
from typing import TypedDict

from . import common, core


class StageCtx(TypedDict):
    body: str
    prev_body: str
    prior_summaries: list[str]
    model: str


def run_episode(seq: int, registry: dict, *, model: str, out_dir: Path) -> None:
    chars = out_dir / "chars"
    summaries = out_dir / "summaries"
    registry_path = chars / "registry.json"

    t0 = time.perf_counter()
    print(f"\n=== ep{seq:03d} ===", flush=True)
    body = common.read_episode(seq)
    prev_body = "" if seq == 1 else common.read_episode(seq - 1)
    prior = core.load_prior_summaries(summaries, before_seq=max(1, seq - 1))
    stem = common.episode_stem(seq)
    ctx: StageCtx = {
        "body": body,
        "prev_body": prev_body,
        "prior_summaries": prior,
        "model": model,
    }

    # extract + verify -----------------------------------------------------
    print("  [extract]", flush=True)
    extract = core.char_extract(**ctx, registry=registry)
    common.save_result(chars / "01_extract", stem, extract)
    names = extract.output

    print("  [extract_verify]", flush=True)
    verify = core.char_extract_verify(body=body, names=names, model=model)
    common.save_result(chars / "02_extract_verify", stem, verify)
    v = verify.output
    if not v["complete"] or v["hallucination"]:
        raise SystemExit(
            f"ep{seq} extract failed verification "
            f"(complete={v['complete']} hallucination={v['hallucination']} "
            f"missing={v['missing']} hallucinated={v['hallucinated']})"
        )

    # merge + apply --------------------------------------------------------
    if not registry:
        print("  [update_reg_a]", flush=True)
        core.update_reg_a(registry, names, seq)
    else:
        print(f"  [merge_eval] {len(names)} targets", flush=True)
        eval_dir = chars / "03_merge_eval" / stem
        evals: dict[str, common.Result] = {}
        for n in names:
            cname = n["canonical_name"]
            r = core.char_merge_eval(
                **ctx, registry=registry, names=names, target_canonical_name=cname
            )
            common.save_result(eval_dir, cname, r)
            evals[cname] = r

        print(f"  [merge_judge] {len(names)} targets", flush=True)
        judge_dir = chars / "04_merge_judge" / stem
        judges: dict[str, common.Result] = {}
        for n in names:
            cname = n["canonical_name"]
            cand_reg = core.char_merge_prune_candidates(registry, evals[cname].output)
            r = core.char_merge_judge(
                **ctx,
                registry=registry,
                candidate_registry=cand_reg,
                target_canonical_name=cname,
            )
            common.save_result(judge_dir, cname, r)
            judges[cname] = r

        new_targets = [
            n for n in names if not core.judge_existing_cnames(judges[n["canonical_name"]].output)
        ]
        print(f"  [merge_prune] {len(new_targets)} new", flush=True)
        prune_dir = chars / "05_merge_prune" / stem
        prunes: dict[str, common.Result] = {}
        for n in new_targets:
            cname = n["canonical_name"]
            r = core.char_merge_prune(
                **ctx, registry=registry, names=names, target_canonical_name=cname
            )
            common.save_result(prune_dir, cname, r)
            prunes[cname] = r

        print("  [update_reg_b]", flush=True)
        core.update_reg_b(
            registry,
            names,
            {k: v.output for k, v in judges.items()},
            {k: v.output for k, v in prunes.items()},
            seq,
        )

        ids = core.touched_ids(registry, seq)
        print(f"  [consolidate] {len(ids)} ids", flush=True)
        cons_dir = chars / "06_consolidate" / stem
        consolidations: dict[str, common.Result] = {}
        for cid in ids:
            r = core.char_consolidate(
                **ctx,
                registry=registry,
                target_id=cid,
                target_surface_forms=core.surface_forms_for(cid, registry, names),
            )
            common.save_result(cons_dir, cid, r)
            consolidations[cid] = r

        print(f"  [update_history] {len(ids)} ids", flush=True)
        hist_dir = chars / "07_update_history" / stem
        histories: dict[str, common.Result] = {}
        for cid in ids:
            r = core.char_update_history(**ctx, registry=registry, target_id=cid)
            common.save_result(hist_dir, cid, r)
            histories[cid] = r

        print("  [update_reg_c]", flush=True)
        core.update_reg_c(
            registry,
            {k: v.output for k, v in consolidations.items()},
            {k: v.output for k, v in histories.items()},
            seq,
        )

    common.dump_json(registry, registry_path)

    # summarize ------------------------------------------------------------
    print("  [summarize]", flush=True)
    summary = core.summarize(**ctx, seq=seq, registry=registry)
    common.save_result(summaries, stem, summary)

    print(
        f"ep{seq:03d} done {time.perf_counter() - t0:.1f}s registry={len(registry)}",
        flush=True,
    )
