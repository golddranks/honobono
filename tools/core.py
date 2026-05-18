"""Pipeline stages and chars-registry data logic.

Two responsibilities live here:
  * LLM stage functions (`char_extract`, `char_extract_verify`, ... `summarize`)
    — each does one generate() call, mutates its Result.output to the parsed
    form, returns the Result. No file IO inside.
  * Non-LLM data logic over the chars registry (`update_reg_a/b/c`,
    `char_merge_prune_candidates`, `load_prior_summaries`) used by the driver
    to fold LLM outputs into persistent state.

Harness utilities (paths, JSON IO, the llama.cpp client, the Result
carrier, save_result) live in common.py.
"""

import json
from pathlib import Path

from . import common, prompts
from .common import Result

BASE_OPTS = {"temperature": 0.0, "num_ctx": 32768}


# --- internal helpers -----------------------------------------------------


def _existing_cnames(registry: dict) -> list[str]:
    return [v["canonical_name"] for v in registry.values() if v.get("canonical_name")]


def _find_name(names: list[dict], cname: str) -> dict:
    for n in names:
        if n["canonical_name"] == cname:
            return n
    raise KeyError(cname)


def _generate_parsed(prompt: str, model: str, fmt: dict, num_predict: int) -> Result:
    """Call generate and json.loads the raw text into Result.output."""
    res = common.generate(
        prompt=prompt, model=model, fmt=fmt, num_predict=num_predict, **BASE_OPTS
    )
    res.output = json.loads(res.output)
    return res


# --- LLM stages -----------------------------------------------------------


def char_extract(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    model: str,
) -> Result:
    """Extract the character list for one episode. Output: list of dicts
    `[{canonical_name, other_names, description}, ...]`."""
    prompt = prompts.char_extract_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
    )
    res = _generate_parsed(prompt, model, prompts.CHAR_EXTRACT_SCHEMA, num_predict=4000)
    res.output = res.output["characters"]
    return res


def char_extract_verify(*, body: str, names: list[dict], model: str) -> Result:
    """Verify the extracted list for omissions and hallucinations against
    the body only — no registry / prev_body / summaries are shown to the
    model on purpose (so it can't accept those sources as evidence that an
    extract entry is a real character in this ep). Output: `{complete,
    hallucination, missing[], hallucinated[]}`."""
    prompt = prompts.char_extract_verify_prompt(body=body, names=names)
    return _generate_parsed(prompt, model, prompts.extract_verify_schema(names), num_predict=1000)


def char_merge_eval(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    model: str,
) -> Result:
    """One target × all-existing-cnames likelihood evaluation. Output:
    `{existing_cname: {likelihood, evidence}}`."""
    prompt = prompts.char_merge_eval_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
        names=names,
        target_canonical_name=target_canonical_name,
    )
    return _generate_parsed(
        prompt,
        model,
        prompts.merge_eval_schema(_existing_cnames(registry)),
        num_predict=4000,
    )


def char_merge_judge(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    candidate_registry: dict,
    target_canonical_name: str,
    model: str,
) -> Result:
    """Per-target final identity decision against the pruned candidate set.
    Output: `{candidate_cname: {evidence, verdict}}` for each candidate plus
    the virtual NEW_CHAR_KEY entry. Use `judge_existing_cnames` to collapse
    to a list of matched existing cnames."""
    prompt = prompts.char_merge_judge_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
        candidate_registry=candidate_registry,
        target_canonical_name=target_canonical_name,
    )
    return _generate_parsed(
        prompt,
        model,
        prompts.merge_judge_schema(_existing_cnames(candidate_registry)),
        num_predict=2000,
    )


def judge_existing_cnames(judge_output: dict) -> list[str]:
    """Candidate canonical_names the judge marked verdict=true (same person
    as the target). Excludes the virtual NEW_CHAR_KEY entry."""
    return [c for c, v in judge_output.items() if c != prompts.NEW_CHAR_KEY and v.get("verdict")]


def char_merge_prune(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    model: str,
) -> Result:
    """Per-new-target keep/drop verdict. Output: `{keep: bool, reason}`."""
    target = _find_name(names, target_canonical_name)
    prompt = prompts.char_merge_prune_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
        names=names,
        target_canonical_name=target_canonical_name,
        target_description=target.get("description", ""),
    )
    return _generate_parsed(prompt, model, prompts.MERGE_PRUNE_SCHEMA, num_predict=1000)


def char_consolidate(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_surface_forms: list[str],
    model: str,
) -> Result:
    """Per-target canonical_name / aliases / description consolidation.
    Output: `{canonical_name, aliases, description}`."""
    prompt = prompts.char_consolidate_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
        target_id=target_id,
        target_entry=registry[target_id],
        target_surface_forms=target_surface_forms,
    )
    return _generate_parsed(prompt, model, prompts.CHAR_CONSOLIDATE_SCHEMA, num_predict=2000)


def char_update_history(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    model: str,
) -> Result:
    """Per-target identity-change history entry. Output: `{new_history, reason}`.
    `new_history` is null when no identity change occurred."""
    prompt = prompts.char_update_history_prompt(
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
        target_id=target_id,
        target_entry=registry[target_id],
    )
    return _generate_parsed(prompt, model, prompts.CHAR_UPDATE_HISTORY_SCHEMA, num_predict=1000)


def summarize(
    *,
    seq: int,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    model: str,
) -> Result:
    """Per-ep 1-2 sentence summary. Output: `summary` string."""
    prompt = prompts.summarize_prompt(
        seq=seq,
        body=body,
        prev_body=prev_body,
        prior_summaries=prior_summaries,
        registry=registry,
    )
    res = _generate_parsed(prompt, model, prompts.SUMMARIZE_SCHEMA, num_predict=1000)
    res.output = res.output["summary"]
    return res


# --- non-LLM data logic over the chars registry ---------------------------
#
# Registry shape:
#   {cid: {canonical_name, aliases, desc, history, seen_in}}
# All mutations happen in these three update_reg_* functions; LLM stages
# never write to the registry directly.


def _new_entry(canonical_name: str, description: str, seq: int) -> dict:
    return {
        "canonical_name": canonical_name,
        "aliases": [canonical_name],
        "desc": description or "",
        "history": [[seq, "初登場"]],
        "seen_in": [],
    }


def _allocate_entry(registry: dict, item: dict, seq: int, max_id: int) -> int:
    """Append a fresh registry entry from an extract `item`. Mutates registry;
    returns the new max_id."""
    max_id += 1
    entry = _new_entry(item["canonical_name"], item.get("description", ""), seq)
    for n in item.get("other_names") or []:
        if n and n not in entry["aliases"]:
            entry["aliases"].append(n)
    entry["seen_in"].append(seq)
    registry[str(max_id)] = entry
    return max_id


def update_reg_a(registry: dict, names: list[dict], seq: int) -> dict:
    """Ep1 only: every extracted character becomes a new registry entry.
    Mutates `registry`; returns it."""
    max_id = max((int(i) for i in registry), default=0)
    for item in names:
        max_id = _allocate_entry(registry, item, seq, max_id)
    return registry


def _cname_to_id(registry: dict) -> dict[str, int]:
    return {v["canonical_name"]: int(cid) for cid, v in registry.items()}


def update_reg_b(
    registry: dict,
    names: list[dict],
    judgements: dict[str, dict],
    prunings: dict[str, dict],
    seq: int,
) -> dict:
    """Ep2+: fold per-target merge judgements + per-new-target prunings.
    Mutates `registry`; returns it.

    `judgements[cname]` is the per-target judge output: a dict of
    `{candidate_cname: {evidence, verdict}}` plus the virtual NEW_CHAR_KEY
    entry. Existing matches = candidate cnames with verdict=true (excl
    NEW_CHAR_KEY), resolved against the registry.

    Per extracted name:
      ≥1 existing match (resolved):
        len 1 : add surface forms to entry.aliases, append seq.
        len >1: joint — append seq to each. Joint surface name is NOT
                added to any individual's aliases.
      0 existing matches (or none resolved):
        prunings[cname].keep == false → drop. Else allocate a fresh id.
    """
    cname_to_id = _cname_to_id(registry)
    max_id = max((int(i) for i in registry), default=0)
    for item in names:
        cname = item["canonical_name"]
        existing = judge_existing_cnames(judgements[cname])
        resolved = [cname_to_id[c] for c in existing if c in cname_to_id]
        if not resolved:
            if not prunings.get(cname, {}).get("keep", True):
                continue
            max_id = _allocate_entry(registry, item, seq, max_id)
            cname_to_id[cname] = max_id
            continue
        joint = len(resolved) > 1
        for rid in resolved:
            cid = str(rid)
            entry = registry[cid]
            if not joint:
                for n in [cname, *(item.get("other_names") or [])]:
                    if n and n not in entry["aliases"]:
                        entry["aliases"].append(n)
            if seq not in entry["seen_in"]:
                entry["seen_in"].append(seq)
    return registry


def update_reg_c(
    registry: dict,
    consolidations: dict[str, dict],
    histories: dict[str, dict],
    seq: int,
) -> dict:
    """Post-consolidate: apply per-id consolidation (rewrites canonical_name,
    aliases, desc) and history (appends an identity-change entry if any).
    Mutates `registry`; returns it."""
    for cid, c in consolidations.items():
        if cid not in registry:
            continue
        entry = registry[cid]
        entry["canonical_name"] = c["canonical_name"]
        entry["aliases"] = list(c["aliases"])
        entry["desc"] = c["description"]
    for cid, h in histories.items():
        if cid not in registry:
            continue
        new_h = h.get("new_history")
        if new_h:
            registry[cid]["history"].append([seq, new_h])
    return registry


# --- merge_eval candidate pruning ----------------------------------------

TIER_MASS = {"likely": 0.9, "possible": 0.6, "unlikely": 0.3, "impossible": 0.1}
TIER_ORDER = ["likely", "possible", "unlikely", "impossible"]
TARGET_COST_PER_VALUE = 4.5


def char_merge_prune_candidates(registry: dict, eval_output: dict) -> dict:
    """Probability-mass tier-based candidate selection.

    Each eval entry (incl. the virtual NEW_CHAR_KEY) maps to a mass:
      likely=0.9, possible=0.6, unlikely=0.3, impossible=0.1.
    For each cumulative tier-set in order (just likely, +possible, +unlikely,
    +impossible), compute cost = #entries included and value = sum_mass /
    total_mass. Pick the option whose cost/value ratio is closest to
    TARGET_COST_PER_VALUE. The NEW_CHAR_KEY entry counts in cost & value
    (it consumes budget) but is excluded from the returned candidate registry.
    """
    buckets: dict[str, list[str]] = {t: [] for t in TIER_ORDER}
    for cname, v in eval_output.items():
        lk = v.get("same_person_as_target")
        if lk in buckets:
            buckets[lk].append(cname)

    total_mass = sum(TIER_MASS[t] * len(buckets[t]) for t in TIER_ORDER)
    if total_mass == 0:
        return {}

    options: list[tuple[int, float, set[str]]] = []
    cum_cost = 0
    cum_mass = 0.0
    cum_chars: set[str] = set()
    for tier in TIER_ORDER:
        if not buckets[tier]:
            continue
        cum_cost += len(buckets[tier])
        cum_mass += TIER_MASS[tier] * len(buckets[tier])
        cum_chars = cum_chars | set(buckets[tier])
        options.append((cum_cost, cum_mass / total_mass, set(cum_chars)))

    # Sort key: (distance from target ratio, -value). Ties go to the more
    # inclusive option (higher value = more mass covered).
    best = min(
        options,
        key=lambda o: (abs((o[0] / o[1]) - TARGET_COST_PER_VALUE), -o[1]),
    )
    chosen = best[2] - {prompts.NEW_CHAR_KEY}
    return {cid: e for cid, e in registry.items() if e.get("canonical_name") in chosen}


# --- summaries loader ----------------------------------------------------


def load_prior_summaries(summaries_dir: Path, before_seq: int) -> list[str]:
    """Read all per-ep summary .json files with seq < before_seq and format
    them as '第N話: 要約' lines, in spine order. Returns [] when none."""
    summaries_dir = Path(summaries_dir)
    if not summaries_dir.exists():
        return []
    lines = []
    for path in sorted(summaries_dir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        s = int(path.name[:3])
        if s >= before_seq:
            continue
        text = common.load_json(path)
        lines.append(f"第{s}話: {text}")
    return lines


# --- helpers for the driver (pipeline.py) --------------------------------


def surface_forms_for(target_id: str, registry: dict, names: list[dict]) -> list[str]:
    """Subset of `names`' canonical+other_names that match the registry
    entry's current aliases (i.e. surface forms attributed to this id by
    update_reg_a/b). Falls back to [canonical_name] if nothing matches."""
    aliases = set(registry[target_id]["aliases"])
    surfaces: set[str] = set()
    for item in names:
        forms = [item["canonical_name"], *(item.get("other_names") or [])]
        if any(f in aliases for f in forms):
            surfaces.update(f for f in forms if f)
    if not surfaces:
        surfaces.add(registry[target_id]["canonical_name"])
    return sorted(surfaces)


def touched_ids(registry: dict, seq: int) -> list[str]:
    """Registry ids whose seen_in includes `seq` (i.e. touched by update_reg_a
    or update_reg_b this episode), sorted by int id."""
    return sorted((cid for cid, e in registry.items() if seq in e.get("seen_in", [])), key=int)
