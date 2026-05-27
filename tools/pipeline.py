"""Per-episode pipeline stage. Driver (main.py) constructs the registry,
picks the model + out_dir, and loops over episodes.

Per ep, top to bottom: extract → verify_quote (mechanical fallback) →
verify_semantics (loop) → verify_missing → (update_reg_initial for ep1, or
merge_eval → merge_judge → merge_prune → update_reg_merge for ep2+) →
consolidate → (update_history for ep2+) → update_reg_semantics → summarize.
The registry mutates in place across eps; file IO happens at clearly
visible save_result / dump_json sites only.

Output layout (under out_dir):
  chars/
    01_extract/<stem>.{json,prompt.txt,meta.json}
    02_verify_quote/<stem>/<cname>.{json,prompt.txt,meta.json}   (fallback)
    03_verify_semantics/<stem>/<cname>.{json,prompt.txt,meta.json}
    04_verify_missing/<stem>.{json,prompt.txt,meta.json}
    05_merge_eval/<stem>/<cname>.{json,prompt.txt,meta.json}    (ep2+)
    06_merge_judge/<stem>/<cname>.{json,prompt.txt,meta.json}   (ep2+)
    07_merge_prune/<stem>/<cname>.{json,prompt.txt,meta.json}   (ep2+)
    08_consolidate/<stem>/<cid>.{json,prompt.txt,meta.json}     (ep2+)
    09_update_history/<stem>/<cid>.{json,prompt.txt,meta.json}  (ep2+)
    10_register/<stem>.json                                     (snapshot)
    registry.json                                               (latest)
  summaries/<stem>.{json,prompt.txt,meta.json}

Each attempt:
  - char_extract (one LLM call, dedup) saved as `<stem>.json` (attempt 1)
    or `<stem>.try-N.json` (attempts 2+).
  - per-name verify_quote on any mechanical-bad evidence_excerpt:
    confirmed-absent entries are dropped, model-corrected quotes
    replace the bad ones, model-insists-without-valid-quote → crash.
    Outputs under `02_verify_quote/<save_stem>/<cname>.{...}`.
  - verify_semantics + classify on the cleaned list.
  - per-entry verify_quote recovery for verify_semantics-bad entries
    (same mechanism as mechanical-bad): drop or replace excerpt.
    Outputs under `02_verify_quote/<save_stem>/<cname>.sem.{...}`.
  - verify_missing on the recovered name list.
  - if verify_missing finds missing → retry the whole extract with
    feedback (kept_names + missing).
Excerpt-level problems never trigger a full re-extract; only genuine
recall gaps (verify_missing) do.

Naming: attempt 1 success leaves all paths unsuffixed. Attempt 1
failure renames everything to `.try-1`. Attempts 2+ write `.try-N`
from the start. Successful winning attempt has `.ok` appended.
"""

import time
from pathlib import Path
from typing import TypedDict

from . import common, core

EXTRACT_MAX_ATTEMPTS = 3


class StageCtx(TypedDict):
    episode_text: str
    prev_episode_text: str
    prior_summaries: list[str]
    model: str


def _rename(chars: Path, from_stem: str, to_stem: str) -> None:
    """Rename per-attempt outputs from `from_stem` to `to_stem`. Touches
    file pairs in 01_extract / 03_verify_missing and subdirs in
    02_verify_semantics / 02_verify_quote. Used to (a) suffix the unsuffixed
    attempt-1 outputs to `.try-1` when attempt-1 fails, and (b) attach
    `.ok` to a winning attempt's stem on success. Attempts 2+ already
    write with `.try-N` from the start, so they only need the `.ok`
    rename."""
    for d in (chars / "01_extract", chars / "04_verify_missing"):
        for ext in ("json", "meta.json", "prompt.txt"):
            src = d / f"{from_stem}.{ext}"
            if src.exists():
                src.rename(d / f"{to_stem}.{ext}")
    for subdir_name in ("02_verify_quote", "03_verify_semantics"):
        d = chars / subdir_name / from_stem
        if d.exists():
            d.rename(chars / subdir_name / to_stem)


def run_episode(seq: int, registry: dict, *, model: str, out_dir: Path) -> None:
    chars = out_dir / "chars"
    summaries = out_dir / "summaries"
    registry_path = chars / "registry.json"

    t0 = time.perf_counter()
    print(f"\n=== ep{seq:03d} ===")
    episode_text = common.read_episode(seq)
    prev_episode_text = "" if seq == 1 else common.read_episode(seq - 1)
    prior = core.load_prior_summaries(summaries, before_seq=max(1, seq - 1))
    stem = common.episode_stem(seq)
    ctx: StageCtx = {
        "episode_text": episode_text,
        "prev_episode_text": prev_episode_text,
        "prior_summaries": prior,
        "model": model,
    }

    # extract + verify with retry loop -------------------------------------
    # One retry level, triggered only by genuine recall gaps. Each attempt:
    #   1. char_extract (one LLM call, dedup) — saved unsuffixed.
    #   2. mechanical substring pre-check — for any evidence_excerpt that
    #      isn't a substring of episode_text, call verify_quote (drop or
    #      replace excerpt).
    #   3. verify_semantics + classify on the cleaned list.
    #   4. per-entry verify_quote recovery on any verify_semantics-bad
    #      entries — drop or replace excerpt, same as the mechanical-bad
    #      path. Bad entries no longer trigger full re-extracts.
    #   5. verify_missing on the recovered name list.
    #   6. if missing is empty → success. Else archive as `.try-N` and
    #      retry with feedback (kept_names + missing).
    # Final disk state:
    #   - Single-attempt success: `<stem>.json` (no decoration).
    #   - Multi-attempt success: `<stem>.try-1.json` ... +
    #     `<stem>.try-N.ok.json` (winner).
    #   - All attempts failed: `<stem>.try-1..MAX.json`, then crash.
    retry_feedback: str | None = None
    names: list[dict] = []
    missing: list[str] = []
    for attempt in range(1, EXTRACT_MAX_ATTEMPTS + 1):
        # Attempt 1 writes to unsuffixed paths so a clean single-attempt
        # success leaves no decoration on disk. Attempts 2+ write directly
        # to `<stem>.try-N` from the start — no after-the-fact archiving
        # of the unsuffixed file.
        save_stem = stem if attempt == 1 else f"{stem}.try-{attempt}"
        if attempt > 1:
            print(f"  [extract] retry {attempt}/{EXTRACT_MAX_ATTEMPTS}")
        else:
            print("  [extract]")
        extract = core.char_extract(**ctx, registry=registry, retry_feedback=retry_feedback)
        common.save_result(chars / "01_extract", save_stem, extract)

        # Per-name mechanical recovery: for each character whose
        # evidence_excerpt isn't a substring of episode_text, ask the model
        # whether the character is in body at all and, if so, supply a
        # corrected quote. Confirmed-absent characters get dropped;
        # corrected ones get their quotes replaced in-place. A model that
        # insists on being in body without a valid corrected quote is a
        # crash. This avoids re-running the whole extract just for
        # fixable per-name quote problems — extract retries are reserved
        # for verify_semantics / verify_missing failures.
        mech_bad = core.extract_quote_mechanical_bad(
            extract.output,
            episode_text=episode_text,
            prev_episode_text=prev_episode_text,
            prior_summaries=prior,
            registry=registry,
        )
        if mech_bad:
            print(f"  [extract mechanical bad] {[c for c, _ in mech_bad]}")
            vquote_dir = chars / "02_verify_quote" / save_stem
            confirmed_drops: list[str] = []
            saved: list[str] = []
            insisting: list[tuple[str, str]] = []
            chars_by_name = {c["canonical_name"]: c for c in extract.output["characters"]}
            for cname, _ in mech_bad:
                entry = chars_by_name[cname]
                r = core.char_extract_verify_quote(
                    episode_text=episode_text,
                    target_canonical_name=cname,
                    target_evidence_excerpt=entry["evidence_excerpt"],
                    model=model,
                )
                common.save_result(vquote_dir, cname, r)
                out = r.output
                if not out["in_body"]:
                    confirmed_drops.append(cname)
                    continue
                corrected = out.get("correct_evidence_excerpt")
                if corrected and core.quote_in_episode_text(corrected, episode_text):
                    entry["evidence_excerpt"] = corrected
                    saved.append(cname)
                else:
                    insisting.append((cname, out["reason"]))

            if insisting:
                raise SystemExit(
                    f"ep{seq} extract: {len(insisting)} character(s) the model insists are "
                    f"in <本文> but provided no valid corrected quote: {insisting}"
                )

            if confirmed_drops:
                print(f"  [verify_quote confirmed drops] {confirmed_drops}")
            if saved:
                print(f"  [verify_quote saved with corrected quote] {saved}")
            extract.output["characters"] = [
                c for c in extract.output["characters"]
                if c["canonical_name"] not in confirmed_drops
            ]
            common.save_result(chars / "01_extract", save_stem, extract)

        names = core.extract_persons(extract.output)
        vsem_dir = chars / "03_verify_semantics" / save_stem
        print(f"  [verify_semantics] {len(names)} targets")
        vsem_results: list[tuple[dict, dict]] = []
        for n in names:
            cname = n["canonical_name"]
            r = core.char_extract_verify_semantics(
                episode_text=episode_text,
                target_canonical_name=cname,
                target_evidence_excerpt=n["evidence_excerpt"],
                model=model,
            )
            common.save_result(vsem_dir, cname, r)
            vsem_results.append((n, r.output))
        names, bad, dropped, splits, collapsed = core.classify_verify_semantics_results(
            vsem_results, episode_text
        )
        if splits:
            print(f"  [verify_semantics splits] {splits}")
        if dropped:
            print(f"  [verify_semantics dropped] {dropped}")
        if collapsed:
            print(f"  [verify_semantics dedup] collapsed: {collapsed}")

        # Per-entry recovery for verify_semantics-bad entries: same model
        # as the mechanical-bad path — call verify_quote, then drop the
        # entry (in_body=false) or replace its excerpt (in_body=true with
        # a valid corrected excerpt). The original (semantics-rejected)
        # excerpt is passed as `rejected_excerpts` so the model is
        # steered away from re-proposing it.
        #
        # Subtlety: weak models (4B) can keep claiming in_body=true while
        # supplying excerpts that refer to a different character (e.g.
        # 神様 → 占い師). The substring check passes but the entry is
        # still bogus. To catch this, the corrected excerpt is run
        # through verify_semantics once more; if it still fails
        # (quote_refers_to_target=false or any other hard-fail), the
        # entry is dropped. One re-verify only — no loop.
        if bad:
            print(f"  [verify_semantics bad] {[c for c, _ in bad]}")
            vquote_dir = chars / "02_verify_quote" / save_stem
            chars_by_name = {c["canonical_name"]: c for c in extract.output["characters"]}
            sem_drops: list[str] = []
            sem_saved: list[str] = []
            sem_insisting: list[tuple[str, str]] = []
            for cname, _reason in bad:
                entry = chars_by_name[cname]
                rejected = entry["evidence_excerpt"]
                r = core.char_extract_verify_quote(
                    episode_text=episode_text,
                    target_canonical_name=cname,
                    target_evidence_excerpt=rejected,
                    rejected_excerpts=[rejected],
                    model=model,
                )
                common.save_result(vquote_dir, f"{cname}.sem", r)
                out = r.output
                if not out["in_body"]:
                    sem_drops.append(cname)
                    continue
                corrected = out.get("correct_evidence_excerpt")
                if not (corrected and core.quote_in_episode_text(corrected, episode_text)):
                    sem_insisting.append((cname, out["reason"]))
                    continue
                # Re-verify the corrected excerpt to catch the
                # "in_body=true but excerpt refers to the wrong
                # character" failure mode.
                recheck = core.char_extract_verify_semantics(
                    episode_text=episode_text,
                    target_canonical_name=cname,
                    target_evidence_excerpt=corrected,
                    model=model,
                )
                common.save_result(vsem_dir, f"{cname}.recheck", recheck)
                persons, drop, fail = core.verify_semantics_resolve(
                    {"canonical_name": cname, "evidence_excerpt": corrected},
                    recheck.output,
                    episode_text,
                )
                # For recovery, only the clean single-person-and-same-name
                # case counts as success. Hard fails (fail set), silent
                # drops (category != 人物, unsplittable collective), or
                # the model deciding the corrected excerpt is a different
                # character / collective → drop the entry.
                if (
                    fail is not None
                    or drop is not None
                    or len(persons) != 1
                    or persons[0]["canonical_name"] != cname
                ):
                    sem_drops.append(cname)
                    continue
                entry["evidence_excerpt"] = corrected
                names.append({
                    "canonical_name": cname,
                    "evidence_excerpt": corrected,
                    "category": entry.get("category", "人物"),
                })
                sem_saved.append(cname)
            if sem_insisting:
                raise SystemExit(
                    f"ep{seq} extract: {len(sem_insisting)} character(s) the model insists "
                    f"are in <本文> but provided no valid corrected quote after verify_semantics "
                    f"rejection: {sem_insisting}"
                )
            if sem_drops:
                print(f"  [verify_semantics→quote drops] {sem_drops}")
            if sem_saved:
                print(f"  [verify_semantics→quote saved with corrected quote] {sem_saved}")
            extract.output["characters"] = [
                c for c in extract.output["characters"]
                if c["canonical_name"] not in sem_drops
            ]
            common.save_result(chars / "01_extract", save_stem, extract)
            bad = []

        print("  [verify_missing]")
        vmiss = core.char_extract_verify_missing(
            episode_text=episode_text, names=names, model=model
        )
        common.save_result(chars / "04_verify_missing", save_stem, vmiss)
        missing = vmiss.output["missing"]

        if not missing:
            if attempt > 1:
                _rename(chars, save_stem, f"{save_stem}.ok")
            break

        # verify_missing surfaced a recall gap. Attempt 1 was written
        # unsuffixed; suffix it now. Attempts 2+ are already at `.try-N`.
        if attempt == 1:
            _rename(chars, stem, f"{stem}.try-1")

        if attempt < EXTRACT_MAX_ATTEMPTS:
            kept_names = [n["canonical_name"] for n in names]
            retry_feedback = core.build_extract_retry_feedback(missing, kept_names)
            print(f"  [extract retry signal] kept={kept_names} missing={missing}")

    if missing:
        raise SystemExit(
            f"ep{seq} extract failed verification after {EXTRACT_MAX_ATTEMPTS} attempts "
            f"(missing={missing})"
        )

    # merge + apply --------------------------------------------------------
    # Per-stage dicts store .output directly (pipeline never needs the
    # surrounding Result; prompts/metas are already on disk from save_result).
    is_ep1 = not registry
    # Snapshot aliases per cid before any merge-time mutation. Used after
    # update_reg_merge to compute the per-entry diff of newly-added
    # aliases this episode — passed to update_history so the model can
    # detect identity events (name reveal, role reveal) that show up
    # only as alias additions.
    aliases_before: dict[str, set[str]] = {
        cid: set(e.get("aliases", [])) for cid, e in registry.items()
    }
    if is_ep1:
        print("  [update_reg_initial]")
        core.update_reg_initial(registry, names, seq)
    else:
        print(f"  [merge_eval] {len(names)} targets")
        eval_dir = chars / "05_merge_eval" / stem
        evals: dict[str, dict] = {}
        for n in names:
            cname = n["canonical_name"]
            r = core.char_merge_eval(
                **ctx, registry=registry, names=names, target_canonical_name=cname
            )
            common.save_result(eval_dir, cname, r)
            evals[cname] = r.output

        print(f"  [merge_judge] {len(names)} targets")
        judge_dir = chars / "06_merge_judge" / stem
        judges: dict[str, dict] = {}
        for n in names:
            cname = n["canonical_name"]
            cand_reg = core.char_merge_prune_candidates(registry, evals[cname])
            r = core.char_merge_judge(
                **ctx,
                registry=registry,
                candidate_registry=cand_reg,
                target_canonical_name=cname,
            )
            common.save_result(judge_dir, cname, r)
            judges[cname] = r.output

        new_targets = core.judge_new_target_names(names, judges)
        print(f"  [merge_prune] {len(new_targets)} new")
        prune_dir = chars / "07_merge_prune" / stem
        prunes: dict[str, dict] = {}
        for n in new_targets:
            cname = n["canonical_name"]
            r = core.char_merge_prune(
                **ctx, registry=registry, names=names, target_canonical_name=cname
            )
            common.save_result(prune_dir, cname, r)
            prunes[cname] = r.output

        # merge_confirm: LLM pushback ONLY when prune proposes a merge into
        # an existing entry that eval ranked unlikely/impossible. The
        # likely/possible cases are accepted mechanically (eval already
        # endorsed them as plausible same-person matches). For the
        # contradiction case the model is shown both stages' verdicts and
        # asked to commit.
        confirms: dict[str, dict] = {}
        contradictions = []
        for n in new_targets:
            cname = n["canonical_name"]
            p = prunes[cname]
            mt = p.get("merge_into")
            if p.get("keep") or not mt:
                continue
            tier = evals.get(cname, {}).get(mt, {}).get("same_person_as_target", "impossible")
            if tier in ("impossible", "unlikely"):
                contradictions.append((n, mt, tier))
        if contradictions:
            print(f"  [merge_confirm] {len(contradictions)} pushback proposals")
            for n, mt, _tier in contradictions:
                cname = n["canonical_name"]
                r = core.char_merge_confirm(
                    **ctx,
                    registry=registry,
                    target_canonical_name=cname,
                    target_evidence_excerpt=n["evidence_excerpt"],
                    proposed_merge_cname=mt,
                )
                common.save_result(prune_dir, f"{cname}.confirm", r)
                confirms[cname] = r.output

        print("  [update_reg_merge]")
        core.update_reg_merge(registry, names, judges, evals, prunes, confirms, seq)

    # consolidate runs every episode — for ep1 it populates the desc /
    # canonical_name / aliases that update_reg_initial left empty; for ep2+ it
    # folds the merge results.
    ids = core.touched_ids(registry, seq)
    print(f"  [consolidate] {len(ids)} ids")
    cons_dir = chars / "08_consolidate" / stem
    consolidations: dict[str, dict] = {}
    for cid in ids:
        r = core.char_consolidate(
            **ctx,
            registry=registry,
            target_id=cid,
            target_surface_forms=core.surface_forms_for(cid, registry, names),
        )
        common.save_result(cons_dir, cid, r)
        consolidations[cid] = r.output

    # update_history only runs for ep2+. On ep1 every entry is 初登場 and
    # there's nothing earlier to compare against; running it would burn LLM
    # calls returning new_history=null.
    histories: dict[str, dict] = {}
    if not is_ep1:
        print(f"  [update_history] {len(ids)} ids")
        hist_dir = chars / "09_update_history" / stem
        for cid in ids:
            new_aliases = [
                a for a in registry[cid].get("aliases", [])
                if a not in aliases_before.get(cid, set())
            ]
            r = core.char_update_history(
                **ctx,
                registry=registry,
                target_id=cid,
                new_aliases_this_episode=new_aliases,
            )
            common.save_result(hist_dir, cid, r)
            histories[cid] = r.output

    print("  [update_reg_semantics]")
    core.update_reg_semantics(registry, consolidations, histories, seq)

    # 10_register: snapshot the post-merge registry state for this episode.
    # Both the rolling registry.json (downstream's current state) and the
    # per-episode snapshot point at the same data — snapshots make the
    # merge stage's mutations inspectable episode-by-episode.
    common.dump_json(registry, registry_path)
    register_dir = chars / "10_register"
    register_dir.mkdir(parents=True, exist_ok=True)
    common.dump_json(registry, register_dir / f"{stem}.json")

    # summarize ------------------------------------------------------------
    print("  [summarize]")
    summary = core.summarize(**ctx, seq=seq, registry=registry)
    common.save_result(summaries, stem, summary)

    print(f"ep{seq:03d} done {time.perf_counter() - t0:.1f}s registry={len(registry)}")
