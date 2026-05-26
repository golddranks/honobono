"""Pipeline stages and chars-registry data logic.

Two responsibilities live here:
  * LLM stage functions (`char_extract`, `char_extract_verify`, ... `summarize`)
    — each does one generate() call, mutates its Result.output to the parsed
    form, returns the Result. No file IO inside.
  * Non-LLM data logic over the chars registry (`update_reg_initial/b/c`,
    `char_merge_prune_candidates`, `load_prior_summaries`) used by the driver
    to fold LLM outputs into persistent state.

Harness utilities (paths, JSON IO, the llama.cpp client, the Result
carrier, save_result) live in common.py.
"""

import json
from pathlib import Path
from typing import NamedTuple

from . import common, prompts
from .common import Result

BASE_OPTS = {"temperature": 0.0, "num_ctx": 32768}


def quote_in_episode_text(quote: str, episode_text: str) -> bool:
    """Whether `quote` appears in `episode_text`, tolerant of newlines.
    A common 4B/8B failure mode is reproducing a quote that spans
    consecutive body sentences with the newline between them dropped.
    The semantic content is correct; the mismatch is purely whitespace,
    so we accept it. Stricter fabrications (composing from disjoint
    fragments, paraphrasing) still fail the check."""
    if quote in episode_text:
        return True
    return quote.replace("\n", "") in episode_text.replace("\n", "")


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
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    model: str,
    retry_feedback: str | None = None,
) -> Result:
    """Extract the character list for one episode. One LLM call, then a
    (canonical_name, evidence_excerpt) dedup. Returns the Result.

    Output is the full parsed dict `{episode_text_mentions, characters}`.
    `characters[i]` is `{canonical_name, evidence_excerpt, category}`;
    callers filter to category == "人物" for downstream stages. The full
    dict preserves the model's per-mention categorization.

    Retry is the caller's responsibility: pipeline does a mechanical
    substring pre-check after this returns, then retries with
    `retry_feedback` set if quotes are bad. Same outer loop handles
    LLM-detected issues from verify_semantics / verify_missing."""
    prompt = prompts.char_extract_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        retry_feedback=retry_feedback,
    )
    res = _generate_parsed(prompt, model, prompts.CHAR_EXTRACT_SCHEMA, num_predict=4000)
    seen: set[tuple[str, str]] = set()
    deduped = []
    for c in res.output["characters"]:
        key = (c["canonical_name"], c["evidence_excerpt"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    res.output["characters"] = deduped
    return res


def extract_persons(extract_output: dict) -> list[dict]:
    """Filter `char_extract` output to entries with category == "人物".
    These are the names the rest of the pipeline operates on."""
    return [c for c in extract_output["characters"] if c["category"] == prompts.PERSON_CATEGORY]


def char_extract_verify_semantics(
    *,
    episode_text: str,
    target_canonical_name: str,
    target_evidence_excerpt: str,
    model: str,
) -> Result:
    """Per-extracted-name validity check against episode_text only. Output:
    `{in_episode_text, is_person, is_single, reason}`. Any false → name is bad and
    should fail the extract. Body-only by design (no registry / summaries /
    prev_episode_text) so the model can't accept those sources as evidence that an
    entry is a real character in this ep."""
    prompt = prompts.char_extract_verify_semantics_prompt(
        episode_text=episode_text,
        target_canonical_name=target_canonical_name,
        target_evidence_excerpt=target_evidence_excerpt,
    )
    return _generate_parsed(prompt, model, prompts.VERIFY_SEMANTICS_SCHEMA, num_predict=500)


def char_extract_verify_missing(*, episode_text: str, names: list[dict], model: str) -> Result:
    """Single completeness check against episode_text only. Output: `{missing[], reason}`.
    Non-empty missing → extract failed."""
    prompt = prompts.char_extract_verify_missing_prompt(episode_text=episode_text, names=names)
    return _generate_parsed(prompt, model, prompts.VERIFY_MISSING_SCHEMA, num_predict=2000)


def char_extract_verify_quote(
    *,
    episode_text: str,
    target_canonical_name: str,
    target_evidence_excerpt: str,
    model: str,
    max_attempts: int = 3,
) -> Result:
    """Per-name mechanical recovery LLM call: given an extract entry
    whose evidence_excerpt isn't a substring of episode_text, asks the
    model whether the target is in body at all (in any form) and, if so,
    to supply a corrected quote.

    Output: `{in_body: bool, correct_evidence_excerpt: str | null, reason}`.
    Pipeline uses the result as:
      in_body=false                       → drop the entry silently.
      in_body=true + valid corrected quote → keep with replaced quote.
      in_body=true + null / invalid quote → crash (insisting case).

    Internal retry loop (up to `max_attempts`): if the model says
    in_body=true but the corrected_quote isn't a substring of
    episode_text, retry with the previous bad correction(s) in the
    prompt as feedback. Common 4B/8B failure mode is concatenating
    across newlines — the feedback prods the model to stay within a
    single contiguous body substring."""
    failed_corrections: list[str] = []
    res: Result | None = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"    [verify_quote internal retry {attempt}/{max_attempts}]", flush=True)
        prompt = prompts.char_extract_verify_quote_prompt(
            episode_text=episode_text,
            target_canonical_name=target_canonical_name,
            target_evidence_excerpt=target_evidence_excerpt,
            failed_corrections=failed_corrections,
        )
        res = _generate_parsed(prompt, model, prompts.VERIFY_QUOTE_SCHEMA, num_predict=500)
        out = res.output
        if not out["in_body"]:
            return res
        corrected = out.get("correct_evidence_excerpt")
        if corrected and quote_in_episode_text(corrected, episode_text):
            return res
        if corrected:
            failed_corrections.append(corrected)
    assert res is not None
    return res


def verify_semantics_resolve(
    name: dict,
    per_name: dict,
    episode_text: str,
) -> tuple[list[dict], str | None, str | None]:
    """Resolve a verify_semantics LLM result for one extracted name.

    Returns (resolved_persons, drop_reason, fail_reason). Exactly one of
    the three meaningful cases holds:
    - resolved_persons non-empty, both reasons None : keep these persons.
    - resolved_persons empty, drop_reason set       : silently drop —
      verify caught an extract over-recall (e.g. extract emitted a non-
      person as 人物; verify recategorized). Logged but pipeline continues.
    - resolved_persons empty, fail_reason set       : hard failure —
      indicates a real bug (quote mismatch, misattribution, model
      confusion). Pipeline crashes for visibility.

    Hard-fail conditions:
      - evidence_excerpt is not a literal substring of episode_text
      - LLM said quote_in_text=false
      - LLM said quote_refers_to_target=false
      - B guard: target canonical_name appears inside split_into

    Silent-drop conditions (verify rescuing extract):
      - is_single=true but category != 人物 (verify says non-person)
      - is_single=false with split_into empty (unsplittable collective)

    Pass-through / expand:
      - is_single=true + category=人物                  → keep as-is.
      - is_single=false + split_into non-empty         → synthesize per-
        individual person entries inheriting the parent's evidence_excerpt.

    Splits are TRUSTED (no per-split LLM re-verification); their
    evidence_excerpt is the parent's, which we've already substring-checked.
    """
    if not quote_in_episode_text(name["evidence_excerpt"], episode_text):
        return [], None, "evidence_excerpt not a substring of episode_text"
    if not per_name["quote_in_text"]:
        return [], None, "quote_in_text=false"
    if not per_name["quote_refers_to_target"]:
        return [], None, "quote_refers_to_target=false"

    if per_name["is_single"]:
        if per_name["category"] != prompts.PERSON_CATEGORY:
            return [], f"category={per_name['category']!r} (not 人物)", None
        return [name], None, None

    if per_name["split_into"]:
        # Mechanical invariant: a collective's components must not include
        # the collective name itself. Seen in practice when the model
        # misreads is_single as "quote has multiple referents" and emits
        # split_into=["母","父"] for canonical_name="母".
        if name["canonical_name"] in per_name["split_into"]:
            return [], None, "split_into contains target canonical_name itself"
        return [
            {
                "canonical_name": s,
                "evidence_excerpt": name["evidence_excerpt"],
                "category": prompts.PERSON_CATEGORY,
            }
            for s in per_name["split_into"]
        ], None, None

    return [], "is_single=false and split_into empty (unsplittable collective)", None


class VerifySemanticsClassification(NamedTuple):
    """Final classification of an episode's verify_semantics LLM outputs.
    Pipeline destructures and uses: resolved (downstream names), bad
    (retry/crash signal), and the three log-only fields."""

    resolved: list[dict]
    bad: list[tuple[str, str]]
    dropped: list[tuple[str, str]]
    splits: dict[str, list[str]]
    collapsed: list[str]


def classify_verify_semantics_results(
    results: list[tuple[dict, dict]],
    episode_text: str,
) -> VerifySemanticsClassification:
    """Take a list of (name, verify_semantics LLM output) pairs and produce
    the buckets the pipeline needs. Two passes:

    1. Per-entry resolve via `verify_semantics_resolve`:
       - keep singles, expand collective splits, accumulate drops/fails.
    2. Dedup by canonical_name across the resolved list (first occurrence
       wins; split-derived duplicates of already-extracted individuals
       collapse).

    `splits` and `collapsed` are pipeline log-only — they don't affect
    downstream data flow, but are useful for spotting model behavior
    quirks (a collective being split; a duplicate being collapsed)."""
    resolved: list[dict] = []
    bad: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []
    splits: dict[str, list[str]] = {}
    for name, per_name in results:
        cname = name["canonical_name"]
        persons, drop, fail = verify_semantics_resolve(name, per_name, episode_text)
        if fail is not None:
            bad.append((cname, fail))
            continue
        if drop is not None:
            dropped.append((cname, drop))
            continue
        if len(persons) > 1 or persons[0]["canonical_name"] != cname:
            splits[cname] = [p["canonical_name"] for p in persons]
        resolved.extend(persons)
    seen: set[str] = set()
    deduped: list[dict] = []
    collapsed: list[str] = []
    for p in resolved:
        cn = p["canonical_name"]
        if cn in seen:
            collapsed.append(cn)
            continue
        seen.add(cn)
        deduped.append(p)
    return VerifySemanticsClassification(deduped, bad, dropped, splits, collapsed)


def judge_new_target_names(names: list[dict], judges: dict[str, dict]) -> list[dict]:
    """Filter `names` to those whose judge output has no matched existing
    cname (merge judge says they're new characters). `judges` maps
    canonical_name → judge LLM output dict (post-`.output` unwrap)."""
    return [n for n in names if not judge_existing_cnames(judges[n["canonical_name"]])]


def extract_quote_mechanical_bad(
    extract_output: dict,
    *,
    episode_text: str,
    prev_episode_text: str = "",
    prior_summaries: list[str] | tuple[str, ...] = (),
    registry: dict | None = None,
) -> list[tuple[str, str]]:
    """Find characters whose evidence_excerpt is not a literal substring of
    `episode_text`. For each one, scan the other context blocks
    (`<前回の本文>`, `<これまでのあらすじ>`, `<既存の人物>`) and produce
    a (canonical_name, advice) pair that — when the source is identified
    — tells the model exactly which block it mistakenly pulled from."""
    other_blocks: list[tuple[str, str]] = []
    if prev_episode_text:
        other_blocks.append(("前回の本文", prev_episode_text))
    if prior_summaries:
        other_blocks.append(("これまでのあらすじ", "\n".join(prior_summaries)))
    if registry:
        other_blocks.append(("既存の人物", prompts.render_registry(registry)))
    bad: list[tuple[str, str]] = []
    for c in extract_output["characters"]:
        quote = c["evidence_excerpt"]
        cname = c["canonical_name"]
        if quote_in_episode_text(quote, episode_text):
            continue
        source = next(
            (block_name for block_name, block_text in other_blocks if quote in block_text),
            None,
        )
        if source:
            advice = (
                f"前回書いた evidence_excerpt「{quote}」は <{source}> 内の文字列です。"
                f"<{source}> から引用してはいけません。"
                f"今回の <本文> を読み直し、{cname} への言及をその中から見つけて、その実在する箇所を引用してください。"
                f"もし {cname} が <本文> 中で全く言及されていなければ、過去の話に出ただけの人物として characters から外してください。"
            )
        else:
            advice = (
                f"前回書いた evidence_excerpt「{quote}」は <本文> 内に存在しません。"
                f"要約・言い換え・連結ではなく、<本文> の中に文字どおりに現れる文字列をそのまま引用してください。"
                f"もし {cname} が <本文> 中で全く言及されていなければ、characters から外してください。"
            )
        bad.append((cname, advice))
    return bad


def build_extract_retry_feedback(
    bad: list[tuple[str, str]],
    missing: list[str],
    kept_names: list[str],
) -> str:
    """Compose a Japanese feedback block for the next extract attempt.
    Three sections:
    - `kept_names`: characters that survived the previous verify pass.
      The next attempt must include them again — the first extract has
      good recall; retries refine specific problems and shouldn't lose
      the good ones.
    - `bad`: per-character advice for problems verify caught
      (mis-attributed quotes, B guard, etc.).
    - `missing`: characters verify_missing flagged as in body but not
      extracted.
    Silent drops are not surfaced (verify already handled them)."""
    lines = ["前回の出力に以下の問題がありました。今回はこれらを必ず直してください。"]
    if kept_names:
        lines.append("")
        lines.append("【前回の出力で正しく抽出できた人物（今回も必ず含めること）】")
        for n in kept_names:
            lines.append(f"- {n}")
    if bad:
        lines.append("")
        lines.append("【evidence_excerpt の問題】")
        for cname, advice in bad:
            lines.append(f"- {cname}: {advice}")
    if missing:
        lines.append("")
        lines.append("【本文中で言及されているのに characters に含まれていなかった人物】")
        for m in missing:
            lines.append(
                f"- 「{m}」は <本文> 中で言及されていますが、前回の characters には含まれていませんでした。"
                f"今回は必ず含めてください。"
            )
    return "\n".join(lines)


def char_merge_eval(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    model: str,
) -> Result:
    """One target × all-existing-cnames likelihood evaluation. Output:
    `{existing_cname: {likelihood, evidence}}`."""
    prompt = prompts.char_merge_eval_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        names=names,
        target_canonical_name=target_canonical_name,
    )
    return _generate_parsed(
        prompt,
        model,
        prompts.merge_eval_schema(_existing_cnames(registry), target_canonical_name),
        num_predict=4000,
    )


def char_merge_judge(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    candidate_registry: dict,
    target_canonical_name: str,
    model: str,
) -> Result:
    """Per-target final identity decision against the pruned candidate set.
    Output: `{candidate_cname: {target, evidence, same_person_as_target}}`
    for each candidate plus the virtual NEW_CHAR_KEY entry. Use
    `judge_existing_cnames` to collapse to a list of matched existing
    cnames."""
    prompt = prompts.char_merge_judge_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        candidate_registry=candidate_registry,
        target_canonical_name=target_canonical_name,
    )
    return _generate_parsed(
        prompt,
        model,
        prompts.merge_judge_schema(_existing_cnames(candidate_registry), target_canonical_name),
        num_predict=2000,
    )


def judge_existing_cnames(judge_output: dict) -> list[str]:
    """Candidate canonical_names the judge marked same_person_as_target=true.
    Excludes the virtual NEW_CHAR_KEY entry."""
    return [
        c for c, v in judge_output.items()
        if c != prompts.NEW_CHAR_KEY and v.get("same_person_as_target")
    ]


_TIER_RANK = {"likely": 3, "possible": 2, "unlikely": 1, "impossible": 0}


def collapse_multi_judge(
    existing_true: list[str],
    eval_output: dict,
    target_cname: str,
) -> list[str]:
    """When merge_judge marks ≥2 existing candidates as same-person as the
    target, the model has hedged: extract guarantees singular targets
    (collectives are split in verify_semantics) so at most one real match
    can exist. Disambiguate by picking the candidate with the highest
    merge_eval likelihood tier. A tie at the top tier (≥2 candidates
    share the highest tier) is a genuine ambiguity the pipeline can't
    silently resolve — raise SystemExit so the run fails loudly."""
    if len(existing_true) <= 1:
        return existing_true

    def tier(c: str) -> int:
        return _TIER_RANK.get(eval_output.get(c, {}).get("same_person_as_target", "impossible"), 0)

    max_tier = max(tier(c) for c in existing_true)
    top = [c for c in existing_true if tier(c) == max_tier]
    if len(top) > 1:
        raise SystemExit(
            f"merge_judge multi-true tie for target {target_cname!r}: "
            f"candidates {top} all share top merge_eval tier "
            f"({[k for k, v in _TIER_RANK.items() if v == max_tier][0]!r}). "
            f"Pipeline cannot pick a winner."
        )
    return top


def char_merge_prune(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    model: str,
) -> Result:
    """Per-new-target keep/drop verdict. Output: `{keep: bool, reason}`."""
    target = _find_name(names, target_canonical_name)
    prompt = prompts.char_merge_prune_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        names=names,
        target_canonical_name=target_canonical_name,
        target_description=target.get("description", ""),
    )
    return _generate_parsed(prompt, model, prompts.MERGE_PRUNE_SCHEMA, num_predict=1000)


def char_consolidate(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_surface_forms: list[str],
    model: str,
) -> Result:
    """Per-target canonical_name / aliases / description consolidation.
    Output: `{canonical_name, aliases, description}`."""
    prompt = prompts.char_consolidate_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        target_id=target_id,
        target_entry=registry[target_id],
        target_surface_forms=target_surface_forms,
    )
    return _generate_parsed(prompt, model, prompts.CHAR_CONSOLIDATE_SCHEMA, num_predict=2000)


def char_update_history(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    model: str,
) -> Result:
    """Per-target identity-change history entry. Output: `{new_history, reason}`.
    `new_history` is null when no identity change occurred."""
    prompt = prompts.char_update_history_prompt(
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
        prior_summaries=prior_summaries,
        registry=registry,
        target_id=target_id,
        target_entry=registry[target_id],
    )
    return _generate_parsed(prompt, model, prompts.CHAR_UPDATE_HISTORY_SCHEMA, num_predict=1000)


def summarize(
    *,
    seq: int,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    model: str,
) -> Result:
    """Per-ep 1-2 sentence summary. Output: `summary` string."""
    prompt = prompts.summarize_prompt(
        seq=seq,
        episode_text=episode_text,
        prev_episode_text=prev_episode_text,
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


def update_reg_initial(registry: dict, names: list[dict], seq: int) -> dict:
    """Ep1 only: every extracted character becomes a new registry entry.
    Mutates `registry`; returns it."""
    max_id = max((int(i) for i in registry), default=0)
    for item in names:
        max_id = _allocate_entry(registry, item, seq, max_id)
    return registry


def _cname_to_id(registry: dict) -> dict[str, int]:
    return {v["canonical_name"]: int(cid) for cid, v in registry.items()}


def update_reg_merge(
    registry: dict,
    names: list[dict],
    judgements: dict[str, dict],
    evals: dict[str, dict],
    prunings: dict[str, dict],
    seq: int,
) -> dict:
    """Ep2+: fold per-target merge judgements + per-new-target prunings.
    Mutates `registry`; returns it.

    `judgements[cname]` is the per-target judge output: a dict of
    `{candidate_cname: {target, evidence, same_person_as_target}}` plus
    the virtual NEW_CHAR_KEY entry. Existing matches = candidate cnames
    with same_person_as_target=true (excl NEW_CHAR_KEY), resolved against
    the registry. If the model hedges with ≥2 true existing candidates,
    `collapse_multi_judge`
    picks the highest-tier one from `evals[cname]` — targets here are
    always singular (collectives are split upstream), so multi-match is
    a model bug, not legitimate joint membership.

    Per extracted name:
      ≥1 existing match (resolved, single after collapse):
        add surface forms to entry.aliases, append seq.
      0 existing matches (or none resolved):
        prunings[cname].keep == false → drop. Else allocate a fresh id.
    """
    cname_to_id = _cname_to_id(registry)
    max_id = max((int(i) for i in registry), default=0)
    for item in names:
        cname = item["canonical_name"]
        existing = collapse_multi_judge(
            judge_existing_cnames(judgements[cname]), evals[cname], cname
        )
        resolved = [cname_to_id[c] for c in existing if c in cname_to_id]
        if not resolved:
            if not prunings.get(cname, {}).get("keep", True):
                continue
            max_id = _allocate_entry(registry, item, seq, max_id)
            cname_to_id[cname] = max_id
            continue
        # `resolved` is guaranteed to have at most one entry after
        # collapse_multi_judge — single match path only.
        cid = str(resolved[0])
        entry = registry[cid]
        for n in [cname, *(item.get("other_names") or [])]:
            if n and n not in entry["aliases"]:
                entry["aliases"].append(n)
        if seq not in entry["seen_in"]:
            entry["seen_in"].append(seq)
    return registry


def update_reg_semantics(
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
TARGET_COST_PER_VALUE = 5.0


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
    update_reg_initial/b). Falls back to [canonical_name] if nothing matches."""
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
    """Registry ids whose seen_in includes `seq` (i.e. touched by update_reg_initial
    or update_reg_merge this episode), sorted by int id."""
    return sorted((cid for cid, e in registry.items() if seq in e.get("seen_in", [])), key=int)
