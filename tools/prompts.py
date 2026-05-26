"""LLM prompts and output JSON schemas for the chars pipeline."""

import json

_RENDER_DROP_KEYS = {"seen_in", "canonical_name"}  # bookkeeping + redundant (now key)
NEW_CHAR_KEY = "新規"


def summaries(prior_summaries: list[str]) -> str:
    """Block-render prior episode summaries, or empty when there are none."""
    if not prior_summaries:
        return ""
    return "\n".join(["\n<これまでのあらすじ>"] + prior_summaries + ["</これまでのあらすじ>\n"])


def prev_body_block(prev_body: str) -> str:
    if not prev_body:
        return ""
    return f"\n<前回の本文>\n{prev_body}\n</前回の本文>\n"


def _slim_entry(entry: dict) -> dict:
    return {
        k: v
        for k, v in entry.items()
        if k not in _RENDER_DROP_KEYS and v is not None and v != "" and v != []
    }


def render_registry(reg: dict) -> str:
    """JSON-render the registry keyed by canonical_name. Returns empty string
    when reg is empty (caller injects nothing into the prompt)."""
    if not reg:
        return ""
    slim = {entry["canonical_name"]: _slim_entry(entry) for entry in reg.values()}
    return "\n<既存の人物>\n" + json.dumps(slim, ensure_ascii=False, indent=1) + "\n</既存の人物>\n"


def render_candidates(candidate_registry: dict) -> str:
    """Render the pruned candidate set keyed by canonical_name, with the
    virtual NEW_CHAR_KEY entry appended. Used by char_merge_judge_prompt as
    the focused decision block (placed right before 本文). The full registry
    goes through render_registry separately as backstory."""
    slim = {entry["canonical_name"]: _slim_entry(entry) for entry in candidate_registry.values()}
    slim[NEW_CHAR_KEY] = {"desc": "既存の人物のいずれでもない、新規の人物"}
    return "\n<候補人物>\n" + json.dumps(slim, ensure_ascii=False, indent=1) + "\n</候補人物>\n"


def render_names(names: list[dict]) -> str:
    return (
        "\n<今回の登場人物>\n"
        + json.dumps(names, ensure_ascii=False, indent=1)
        + "\n</今回の登場人物>\n"
    )


# ---------------------------------------------------------------- extract ---


def char_extract_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
) -> str:
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    return f"""登場人物のリスト作成。

目的は、本文中で人物として扱われている存在を漏れなく列挙することです。

ルール:
- 本文中で行動・発言・思考・知覚している人物を書く。
- その場にいなくても、会話・回想・説明の中で言及された人物も書く。
- 「店員」「母親」など、固有名がなかったり、明示的に呼ばれなくても、特定の人物なら含める。
- 一人称の場合、語り手を忘れずに。
- 同一人物かどうか分からない別名・呼び方の場合、その名前で別々に書いてよい。
- 人物かどうか迷う場合は含めてよい。
- 人間・人型存在・人格を持つ存在のみ。世界・場所・国・物・抽象概念は含めない。
- 一人ずつ列挙する。「両親」「兄と姉」「兄弟」のように複数人を同時に指す呼び方は、分けて書く。分けられない場合は含めない。
- canonical_name は今回の本文中で実際に使われている呼び方から選ぶ。<既存の人物>に出てくる名前を、本文中に登場しないのに流用してはならない。
- evidence_quote は本文中の文字列をそのまま引用する（要約・言い換え禁止）。判定対象の人物を指している短い箇所を選ぶ。50文字以内が目安。

<既存の人物>・<これまでのあらすじ>・<前回の本文> は文脈として表示しているだけ。
今回の本文に登場しない人物を、それらを根拠に含めてはならない。
{plot}{reg_block}{prev}
<本文>
{body}
</本文>

JSONのみを出力すること。説明文・Markdown・コードフェンスは禁止。形式は以下のように。
{{
    "characters": [
        {{
            "canonical_name": "本文中で最も代表的な呼び方",
            "evidence_quote": "本文中からのそのままの引用"
        }}
    ]
}}
"""


CHAR_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string", "maxLength": 100},
                    "evidence_quote": {"type": "string", "maxLength": 200},
                },
                "required": ["canonical_name", "evidence_quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["characters"],
    "additionalProperties": False,
}


# --------------------------------------------------------- extract_verify ---
#
# Two-step verify, both body-only (no registry / summaries / prev_body):
#   verify_each   — loop per extracted name, three binary checks
#   verify_missing — single body scan for names the extract dropped


def char_extract_verify_each_prompt(
    *,
    body: str,
    target_canonical_name: str,
    target_evidence_quote: str,
) -> str:
    target_block = (
        "\n<判定対象>\n"
        f"canonical_name: {target_canonical_name}\n"
        f"evidence_quote: {target_evidence_quote}\n"
        "</判定対象>\n"
    )
    return f"""人物候補の検証（一人ずつ）。

「判定対象」が以下の3点をすべて満たすかを、本文のみを根拠に判定してください。

- in_body : 「判定対象」が本文中に登場する（行動・発言・思考・言及のいずれか）。
  evidence_quote と一致する文字列が本文中に実在し、かつそれが判定対象の人物を指していること。
  本文に存在しない引用や、別人を指す引用なら false。
- is_person : 人間・人型存在・人格を持つ存在である。
  世界・場所・国・村・物・抽象概念・群衆の総称は false。神・精霊など人格を持つ存在は true。
- is_single : 一人の特定の人物を指す。
  「両親」「兄と姉」「兄弟」「村人たち」など複数を同時に指す呼び名は false。

ルール:
- 本文のみを根拠とする。判定対象に関する事前知識・推測・既存情報は使わない。
- reason に判定の根拠を短く書く（false がある場合はどれが false でなぜか）。
{target_block}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "in_body": true または false,
    "is_person": true または false,
    "is_single": true または false,
    "reason": "判定の根拠"
}}
"""


VERIFY_EACH_SCHEMA = {
    "type": "object",
    "properties": {
        "in_body": {"type": "boolean"},
        "is_person": {"type": "boolean"},
        "is_single": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["in_body", "is_person", "is_single", "reason"],
    "additionalProperties": False,
}


def char_extract_verify_missing_prompt(
    *,
    body: str,
    names: list[dict],
) -> str:
    names_block = render_names(names)
    return f"""登場人物リストの漏れ検出。

「今回の登場人物」リストに、本文に登場するが含まれていない人物がいないかを、本文のみを根拠に確認してください。

ルール:
- 本文中で行動・発言・思考・知覚している人物がリストに含まれているか確認する。
- 会話・回想・説明の中で言及された人物（その場にいなくても）がリストに含まれているか確認する。
- 固有名がなくても、特定の人物として識別できる存在（「店員」「母親」など）も対象。
- 一人称の語り手を忘れないこと。
- 人間・人型存在・人格を持つ存在のみが対象。世界・場所・国・物・抽象概念は含めない。
- 「両親」「兄と姉」のような複数を同時に指す呼び方は、分けて missing に書く（個別に分けられないなら含めない）。
- 既に「今回の登場人物」に含まれている人物は missing に入れない。
- 漏れている人物の呼び名を missing に列挙。漏れがなければ空配列。
{names_block}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "missing": ["漏れている人物名", ...],
    "reason": "判定の根拠"
}}
"""


VERIFY_MISSING_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
        },
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["missing", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------- merge_eval -


def char_merge_eval_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
) -> str:
    """Per-target likelihood-of-match evaluation against every registry entry,
    plus a virtual "新規" entry for "target is a new character not in registry"."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    names_block = render_names(names)
    return f"""人物の同一性候補評価。

本文の中の「{target_canonical_name}」が「既存の人物」のそれぞれと同じ人物である可能性を、証拠に基づいて評価してください。
加えて、対象が「既存の人物」のいずれでもない『新規の人物』である可能性も同じ尺度で評価してください（キーは「{NEW_CHAR_KEY}」とする）。

same_person_as_target の意味:
- "likely"     : 同一人物（または「{NEW_CHAR_KEY}」の場合は新規人物）だと考える強い証拠がある
- "possible"   : 同一（または新規）の可能性はあるが決定打に欠ける
- "unlikely"   : 一応排除できないが同一（または新規）の可能性は低い
- "impossible" : 明らかに別人物（または「{NEW_CHAR_KEY}」の場合は明らかに既存人物のいずれか）

ルール:
- 既存の人物それぞれについて evidence と same_person_as_target を書く。
- 加えて「{NEW_CHAR_KEY}」キーで、対象が新規人物である可能性を同じ形式で評価する。
- evidence は本文・前回の本文・あらすじ・既存の人物情報を根拠にする。
- 推測ではなく、明示的な手がかり（呼び名・役割・関係・状況）を引用する。
- 「私」など語り手の一人称は、本文の語り手が誰なのかを慎重に考える。
{plot}{reg_block}{prev}{names_block}
<本文>
{body}
</本文>

対象: 「{target_canonical_name}」

JSONのみを出力すること。形式は以下のように。
{{
    "既存人物の canonical_name または「{NEW_CHAR_KEY}」": {{
        "evidence": "本文や既存情報からの具体的な手がかり",
        "same_person_as_target": "likely" | "possible" | "unlikely" | "impossible"
    }}
}}
"""


LIKELIHOOD_ENUM = ["likely", "possible", "unlikely", "impossible"]


def merge_eval_schema(existing_cnames: list[str]):
    """{cname: {same_person_as_target, evidence}} for every existing cname
    plus the virtual NEW_CHAR_KEY entry."""
    item = {
        "type": "object",
        "properties": {
            "evidence": {"type": "string", "maxLength": 800},
            "same_person_as_target": {"type": "string", "enum": LIKELIHOOD_ENUM},
        },
        "required": ["same_person_as_target", "evidence"],
        "additionalProperties": False,
    }
    keys = [*existing_cnames, NEW_CHAR_KEY]
    return {
        "type": "object",
        "properties": dict.fromkeys(keys, item),
        "required": keys,
        "additionalProperties": False,
    }


# --------------------------------------------------------------- merge_judge


def char_merge_judge_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    candidate_registry: dict,
    target_canonical_name: str,
) -> str:
    """Final per-target identity decision against a pruned candidate set.
    Layout: 要約 → 既存の人物 (full backstory) → 前回の本文 → 候補人物 (+ 新規)
    → 判定対象 → 本文. The focused decision blocks sit right before 本文 so
    the model has them in immediate context when reading the current text."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    cand_block = render_candidates(candidate_registry)
    target_block = f"\n<判定対象>\n{target_canonical_name}\n</判定対象>\n"
    return f"""人物の同一性判定。

「判定対象」が「候補人物」のそれぞれと同じ人物かを、候補ごとに真偽で判断してください。
「{NEW_CHAR_KEY}」は「候補人物のいずれでもない、新規の人物である」可能性を表します。

ルール:
- 候補人物それぞれについて evidence と verdict を書く。
- verdict は true（判定対象と同一人物だと思う）または false（別人だと思う）。
- evidence は本文・前回の本文・あらすじ・既存の人物情報を根拠にする。明示的な手がかり（呼び名・役割・関係・状況）を引用する。
- 「両親」など複数の人物を同時に指す呼び名は、該当する全員に verdict=true をつける。
- 「{NEW_CHAR_KEY}」の verdict は「候補人物のいずれでもない新規の人物だ」と思うなら true。
- 「私」など語り手の一人称は、本文の語り手その人が誰なのかを慎重に考える。
{plot}{reg_block}{prev}{cand_block}{target_block}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "候補のcanonical_name または「{NEW_CHAR_KEY}」": {{
        "evidence": "判定の根拠",
        "verdict": true または false
    }}
}}
"""


def merge_judge_schema(candidate_cnames: list[str]):
    """Per-candidate {evidence, verdict} for each candidate cname plus the
    virtual NEW_CHAR_KEY entry. Downstream collapses verdicts to existing_ids
    (cnames with verdict=true, excl NEW_CHAR_KEY)."""
    item = {
        "type": "object",
        "properties": {
            "evidence": {"type": "string", "maxLength": 500},
            "verdict": {"type": "boolean"},
        },
        "required": ["evidence", "verdict"],
        "additionalProperties": False,
    }
    keys = [*candidate_cnames, NEW_CHAR_KEY]
    return {
        "type": "object",
        "properties": dict.fromkeys(keys, item),
        "required": keys,
        "additionalProperties": False,
    }


# --------------------------------------------------------------- merge_prune


def char_merge_prune_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    target_description: str,
) -> str:
    """For a newly-extracted character (after merge said 'new'), decide if it
    is actually a character worth tracking, or extraction noise."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    names_block = render_names(names)
    return f"""新規登録人物の妥当性判定。

判定対象は今回新しく抽出された人物候補です。これが追跡する価値のある具体的な人物かを判断してください。

ルール:
- 本文に登場する具体的な個人（固有・特定可能な存在）なら keep=true。
- 一般化された群や役割の総称、本文内で語られる物語の中の登場人物、
  一過性の言及で追跡する価値がないものは keep=false。
- 既存の人物と重複する単なる別呼称が抽出ミスとして残った場合も keep=false。
- reason に短く根拠を述べる。
{plot}{reg_block}{prev}{names_block}
<本文>
{body}
</本文>

判定対象: 「{target_canonical_name}」
判定対象の説明: {target_description}

JSONのみを出力すること。形式は以下のように。
{{
    "keep": true または false,
    "reason": "判定の根拠"
}}
"""


MERGE_PRUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["keep", "reason"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- consolidate


def char_consolidate_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_entry: dict,
    target_surface_forms: list[str],
) -> str:
    """Per-target alias/canonical_name/description consolidation."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    target_block = (
        "\n<整理対象>\n"
        + json.dumps(
            {
                "id": target_id,
                "current": {
                    "canonical_name": target_entry.get("canonical_name", ""),
                    "aliases": target_entry.get("aliases", []),
                    "desc": target_entry.get("desc", ""),
                },
                "new_surface_forms_this_episode": target_surface_forms,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n</整理対象>\n"
    )
    return f"""人物情報の整理（一人ずつ）。

整理対象の人物について、これまでの情報と今回のエピソードの内容を統合し、
正規の呼び方・別名・説明を整理してください。

ルール:
- canonical_name: 本文・既存情報を踏まえて最も代表的な呼び方を一つ選ぶ。
  本名が判明していれば本名を、未判明なら役割呼び（例「占い師」）でよい。
- aliases: 意味のある呼び名のみ残す。表記揺れ・単なる重複は除く。
  本名・愛称・役職呼びなど、実質的に異なる呼び方は残す。
  canonical_name 自体も aliases に含める。
- description: その人物を表す一つの短い説明文を書く（150文字以内）。
  最新の情報を反映し、矛盾は最新側を優先する。
{plot}{reg_block}{prev}{target_block}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "canonical_name": "代表的な呼び方",
    "aliases": ["呼び方1", "呼び方2"],
    "description": "短い説明"
}}
"""


CHAR_CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string", "maxLength": 100},
        "aliases": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "minItems": 1,
            "maxItems": 30,
        },
        "description": {"type": "string", "maxLength": 500},
    },
    "required": ["canonical_name", "aliases", "description"],
    "additionalProperties": False,
}


# ------------------------------------------------------------- update_history


def char_update_history_prompt(
    *,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_entry: dict,
) -> str:
    """Per-target identity-change history entry. Only identity changes —
    not story events. Output may be null."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    target_block = (
        "\n<対象人物>\n"
        + json.dumps(
            {
                "id": target_id,
                "canonical_name": target_entry.get("canonical_name", ""),
                "aliases": target_entry.get("aliases", []),
                "desc": target_entry.get("desc", ""),
                "history": target_entry.get("history", []),
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n</対象人物>\n"
    )
    return f"""人物の同一性に関する変化の記録（一人ずつ）。

対象人物について、今回のエピソード内で「同一性に関する変化」が起きたかを判定し、
あれば一文で記録してください。

「同一性に関する変化」とは:
- 本名・正体が判明した（例：「占い師」の本名が「アイビー」と判明）
- 性別・年齢・種族などの基本属性が判明した
- 家族関係・所属・職業が判明した（例：「兄」が実は義兄と判明）
- 新しい呼び名・あだ名が付いた・あるいは正式名称が決まった
- 変装・改名した

「同一性に関する変化」ではない（含めない）:
- 物語の出来事・行動・移動・遭遇・戦闘
- 単なる状況描写・感情
- 初登場（紹介は description が担う）
- 同じ情報の繰り返し

ルール:
- 同一性に関する変化があれば new_history に一文で記録（80文字以内）。
- なければ new_history は null。
- 該当する変化が複数ある場合は最も重要な一つだけ。
- reason に判定の根拠（変化の根拠 / なぜ無いのか）を短く書く。
{plot}{reg_block}{prev}{target_block}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "new_history": "同一性に関する変化（無ければ null）",
    "reason": "判定の根拠"
}}
"""


CHAR_UPDATE_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "new_history": {"oneOf": [{"type": "string", "maxLength": 200}, {"type": "null"}]},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["new_history", "reason"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------- summarize


def summarize_prompt(
    *,
    seq: int,
    body: str,
    prev_body: str,
    prior_summaries: list[str],
    registry: dict,
) -> str:
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_body_block(prev_body)
    return f"""エピソードの要約。

今回のエピソード（第{seq}話）の出来事を一文か二文で要約してください。

ルール:
- 物語の進行に関わる主要な出来事のみ書く。
- 既知の人物・状況の繰り返しは避ける。
- 一文か二文（最大で二文）。100文字以内。
- 主観・感想・解説は書かない。事実のみ。
{plot}{reg_block}{prev}
<本文>
{body}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "summary": "要約の文字列"
}}
"""


SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "maxLength": 300}},
    "required": ["summary"],
    "additionalProperties": False,
}
