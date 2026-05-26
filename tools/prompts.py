"""LLM prompts and output JSON schemas for the chars pipeline."""

import json

_RENDER_DROP_KEYS = {"seen_in", "canonical_name"}  # bookkeeping + redundant (now key)
NEW_CHAR_KEY = "新規"

# Per-target category, used by extract + verify_semantics + verify_missing.
# Enum-constrained so the weak model has to commit per entry instead of
# defaulting to "is_person=true". Downstream treats only "人物" as a real
# character; the other buckets exist so the model has a place to put
# obvious non-persons (worlds, skills, abstract states) honestly.
PERSON_CATEGORY = "人物"
CATEGORY_ENUM = [
    PERSON_CATEGORY,
    "場所世界",
    "物道具",
    "スキル能力職業役職",
    "概念状態属性",
    "群衆総称",
    "その他",
]


def summaries(prior_summaries: list[str]) -> str:
    """Block-render prior episode summaries, or empty when there are none."""
    if not prior_summaries:
        return ""
    return "\n".join(["\n<これまでのあらすじ>"] + prior_summaries + ["</これまでのあらすじ>\n"])


def prev_episode_text_block(prev_episode_text: str) -> str:
    if not prev_episode_text:
        return ""
    return f"\n<前回の本文>\n{prev_episode_text}\n</前回の本文>\n"


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
    # Strip `category`: every name here is already filtered to 人物 by
    # extract_persons, so the field is constant noise in downstream prompts.
    slim = [{k: v for k, v in n.items() if k != "category"} for n in names]
    return (
        "\n<今回の登場人物>\n"
        + json.dumps(slim, ensure_ascii=False, indent=1)
        + "\n</今回の登場人物>\n"
    )


# ---------------------------------------------------------------- extract ---


def char_extract_prompt(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    retry_feedback: str | None = None,
) -> str:
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
    # Retry feedback block: when pipeline retries extract after verify_semantics
    # or verify_missing surfaced problems, embed the structured failure
    # info right before the output template so the model has it as the
    # most recent context when generating.
    feedback_block = (
        f"\n<前回の出力の問題点>\n{retry_feedback}\n</前回の出力の問題点>\n"
        if retry_feedback
        else ""
    )
    return f"""登場人物のリスト作成。

目的: 本文に登場する人物を漏れなく列挙する。

含める対象（迷ったら含める）:
- 本文中で行動・発言・思考・知覚している人物。
- その場にいなくても、会話・回想・説明・地の文で言及された人物。
- 「店員」「母親」など、固有名がなくても特定の人物として識別できる存在。
- 一人称の語り手も忘れずに。
- 同一人物かどうか分からない別名・呼び方は、その名前で別エントリとして書いてよい。

複数人を同時に指す呼称（「両親」「兄と姉」「村人たち」など）は、個別の人物として分けて書く（同じ evidence_excerpt を使ってよい）。分けられない不特定多数は含めない。

各エントリのフィールド:
- canonical_name : 本文中で実際に使われている呼び方から選ぶ。<既存の人物>に出てくる名前を、本文中に登場しないのに流用しない。
- evidence_excerpt : <本文> から、その人物への言及を含む部分をそのまま抜粋する（地の文でもセリフでもよい）。重要なのは「その人物を指している箇所」であり、「その人物の発言」ではない。<既存の人物>・<これまでのあらすじ>・<前回の本文> からの抜粋は禁止。要約・言い換え禁止、50文字以内目安。
- category : 以下から一つ。evidence_excerpt の文脈で判定する。
  人物 / 場所世界 / 物道具 / スキル能力職業役職 / 概念状態属性 / 群衆総称 / その他
  characters には基本「人物」のみを含める。人物以外を入れてしまったら正直に分類する（後で除外）。

<既存の人物>・<これまでのあらすじ>・<前回の本文> は文脈として表示しているだけ。
今回の本文に登場しない人物を、それらを根拠に含めてはならない。

出力手順:
(1) 本文を頭から最後まで読み、人物への言及（固有名・役職呼び・関係呼び・「私」「俺」「僕」などの一人称を含む代名詞）を全て episode_text_mentions に列挙する。同じ呼び名は一度だけ書く。
(2) 各 mention について characters エントリを作り、category を付ける。
{plot}{reg_block}{prev}
<本文>
{episode_text}
</本文>

{feedback_block}
JSONのみを出力すること。説明文・Markdown・コードフェンスは禁止。形式例（値は例示で、実際の本文に応じた内容を書く）:
{{
    "episode_text_mentions": ["本文中の呼び名1", "本文中の呼び名2", "（実際の本文に登場する人数だけ）"],
    "characters": [
        {{"canonical_name": "本文中の代表呼称1", "evidence_excerpt": "本文中の引用1", "category": "人物"}},
        {{"canonical_name": "本文中の代表呼称2", "evidence_excerpt": "本文中の引用2", "category": "人物"}}
    ]
}}
"""


CHAR_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_text_mentions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 50,
        },
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string", "maxLength": 100},
                    "evidence_excerpt": {"type": "string", "maxLength": 200},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                },
                "required": ["canonical_name", "evidence_excerpt", "category"],
                "additionalProperties": False,
            },
            "maxItems": 50,
        },
    },
    "required": ["episode_text_mentions", "characters"],
    "additionalProperties": False,
}


# --------------------------------------------------------- extract_verify ---
#
# Two-step verify, both restricted to the current episode_text (no registry /
# summaries / prev_episode_text):
#   verify_semantics — loop per extracted name; checks referent / category / single
#   verify_missing   — single episode-text scan for names the extract dropped


def char_extract_verify_semantics_prompt(
    *,
    episode_text: str,
    target_canonical_name: str,
    target_evidence_excerpt: str,
) -> str:
    target_block = (
        "\n<判定対象>\n"
        f"canonical_name: {target_canonical_name}\n"
        f"evidence_excerpt: {target_evidence_excerpt}\n"
        "</判定対象>\n"
    )
    return f"""人物候補の検証（一人ずつ）。

「判定対象」について、本文のみを根拠に以下を順に判定してください。

- quote_in_text : evidence_excerpt と一致する文字列（部分一致でなく完全一致）が本文中に存在する場合 true。
  本文に書かれていない文や、要約・言い換えされた引用なら false。

- quote_refers_to_target : 上で見つけた evidence_excerpt の中で、判定対象が実際に言及されている場合 true。
  名前そのもの・代名詞・指示語・役職・関係などで判定対象を指していること。
  引用は本文に存在するが、引用の中で別の人物・対象を指している場合は false。
  （例: 「『こんにちは』」が本文にあっても、それが占い師の発言で、判定対象が「母」の場合 → false）

- category : evidence_excerpt の文脈で判定対象が何を指しているかを、以下から一つ選ぶ。
    ・人物                : 一人の人間・人型存在・神・精霊など、人格を持つ個別の存在
    ・場所世界            : 世界・国・地方・村・街・建物・場所
    ・物道具              : 物・道具・食べ物・身につけるもの
    ・スキル能力職業役職  : スキル・能力・職業名そのもの・役職名そのもの
    ・概念状態属性        : 抽象概念・状態・属性・現象・体系
    ・群衆総称            : 複数人を同時に指す総称（村人たち、両親、兄弟）
    ・その他              : 上記のいずれにも該当しない
  語が複数の意味を持つ場合（「占い師」は職業でも特定の人物の呼び名でもある）、
  evidence_excerpt の文脈で一人の人物を指していれば「人物」と判断する。
  能力・スキル・職業名そのものが言及されている場合は、たとえそれを所有する人物がいても「スキル能力職業役職」とする。

- is_single : 判定対象（canonical_name）が一人の特定の人物を指す呼び方なら true。
  canonical_name 自体が複数を同時に指す呼び名（「両親」「兄と姉」「村人たち」など）の場合のみ false。
  evidence_excerpt の文中に他の人物名が一緒に書かれていても判定基準にしない。
  例: canonical_name="母"、evidence_excerpt="母と父の顔を見る" → is_single=true（canonical_name は「母」単独）。
  例: canonical_name="両親"、evidence_excerpt="両親が私を見ない" → is_single=false（canonical_name 自体が複数を指す）。

- split_into : is_single=false の場合のみ、canonical_name が指す複数人の個別の構成員の呼び名を列挙する。
  例: canonical_name="両親" なら split_into=["母", "父"]。canonical_name="兄と姉" なら split_into=["兄", "姉"]。
  個別を識別できない不特定多数（「村人たち」のような匿名の集団）の場合は空配列 []。
  is_single=true の場合は常に空配列 []。

ルール:
- 本文のみを根拠とする。判定対象に関する事前知識・推測・既存情報は使わない。
- reason に判定の根拠を短く書く（false / 人物以外がある場合はそれぞれの根拠も）。
{target_block}
<本文>
{episode_text}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "quote_in_text": true または false,
    "quote_refers_to_target": true または false,
    "category": "人物" | "場所世界" | "物道具" | "スキル能力職業役職" | "概念状態属性" | "群衆総称" | "その他",
    "is_single": true または false,
    "split_into": ["構成員1", "構成員2"] または [],
    "reason": "判定の根拠"
}}
"""


VERIFY_SEMANTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "quote_in_text": {"type": "boolean"},
        "quote_refers_to_target": {"type": "boolean"},
        "category": {"type": "string", "enum": CATEGORY_ENUM},
        "is_single": {"type": "boolean"},
        "split_into": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 10,
        },
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": [
        "quote_in_text",
        "quote_refers_to_target",
        "category",
        "is_single",
        "split_into",
        "reason",
    ],
    "additionalProperties": False,
}


def char_extract_verify_missing_prompt(
    *,
    episode_text: str,
    names: list[dict],
) -> str:
    names_block = render_names(names)
    return f"""登場人物リストの漏れ検出。

「今回の登場人物」リストに、本文に登場するが含まれていない人物がいないかを、本文のみを根拠に確認してください。

判定手順:
(1) まず episode_text_mentions に、本文中で言及・行動・発言・思考されている対象を全て列挙する。
    人物への言及は、固有名・役職呼び・関係呼び・「私」「俺」「僕」などの一人称を含む代名詞、すべての形式を含む。人物以外（場所・物・概念など）も含む。
    本文を頭から最後まで読み、漏れなく列挙する。同じ呼び名は一度だけ書く。各エントリには name と category を付ける。
(2) episode_text_mentions のうち category="人物" のものだけを抽出する。
(3) その「人物」エントリのうち、「今回の登場人物」に含まれていないものだけを missing に書く。
    表記揺れ・別名でも、同一人物が「今回の登場人物」にあれば missing に入れない。
(4) reason に判定の根拠を一文で。

category は以下から一つ選ぶ:
- 人物                : 一人の人間・人型存在・神・精霊など、人格を持つ個別の存在
- 場所世界            : 世界・国・地方・村・街・建物・場所
- 物道具              : 物・道具・食べ物・身につけるもの
- スキル能力職業役職  : スキル・能力・職業名そのもの・役職名そのもの
- 概念状態属性        : 抽象概念・状態・属性・現象・体系
- 群衆総称            : 複数人を同時に指す総称（村人たち、両親、兄弟）
- その他              : 上記のいずれにも該当しない

複数人を同時に指す呼称（「両親」「兄と姉」「兄弟」「村人たち」など）を本文中で見かけた場合:
- 個別の人物として分けて episode_text_mentions に書く（「人物」カテゴリ）。
  例: 本文に「両親」とあれば、母（人物）と父（人物）を別々に書く。
  例: 本文に「兄と姉」とあれば、兄（人物）と姉（人物）を別々に書く。
  分けた個別の人物が「今回の登場人物」に含まれていなければ missing に入れる。
- 個別を識別できない不特定多数（「村人たち」のような匿名の集団）の場合のみ、category=群衆総称 として
  1 エントリで書く（その場合 missing には入れない）。

漏れがなければ missing は空配列。
{names_block}
<本文>
{episode_text}
</本文>

JSONのみを出力すること。形式は以下のように（値はあくまで形式の例で、実際の本文に応じた内容を書く）:
{{
    "episode_text_mentions": [
        {{"name": "本文中の対象の呼び名1", "category": "人物"}},
        {{"name": "本文中の対象の呼び名2", "category": "人物以外のカテゴリも可"}}
    ],
    "missing": ["今回の登場人物に含まれていない人物の呼び名"],
    "reason": "判定の根拠"
}}
"""


VERIFY_MISSING_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_text_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 100},
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                },
                "required": ["name", "category"],
                "additionalProperties": False,
            },
            "maxItems": 50,
        },
        "missing": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 20,
        },
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["episode_text_mentions", "missing", "reason"],
    "additionalProperties": False,
}


# ---------------------------------------------------------- verify_quote ---
#
# Fallback verification, invoked by the pipeline only when extract retries
# have exhausted with mechanically-bad evidence_excerpts (a quote that's not
# a substring of <本文>). For each such character the model is asked, with
# body-only context, whether the character is in fact mentioned in this
# episode in any form. A `false` answer authorizes the pipeline to drop
# that entry silently; a `true` answer is a hard failure (the model claims
# the character is there but couldn't quote it correctly).


def char_extract_verify_quote_prompt(
    *,
    episode_text: str,
    target_canonical_name: str,
    target_evidence_excerpt: str,
    failed_corrections: list[str] | tuple[str, ...] = (),
    rejected_excerpts: list[str] | tuple[str, ...] = (),
) -> str:
    """Per-name "is this character actually in body?" verification.
    Invoked when the previous evidence_excerpt was rejected — either
    mechanically (not a substring of body) or semantically (doesn't
    refer to the target). The prompt is agnostic about which: it just
    asks the model to recheck the target's presence and supply a
    correct excerpt or admit absence.

    `failed_corrections` carries excerpts that the internal retry loop
    has already rejected as non-substrings. `rejected_excerpts` carries
    excerpts the caller already knows are inadequate (typically the
    original extract excerpt that triggered this verification). Both
    are surfaced to the model so it doesn't repeat the same answer."""
    blocks: list[str] = []
    if rejected_excerpts:
        bullets = "\n".join(f"- 「{q}」" for q in rejected_excerpts)
        blocks.append(
            "前回の判定で以下の evidence_excerpt は不適切と判定されました"
            "（本文に存在しない、または対象を指していない、もしくは捏造の可能性）:\n"
            f"{bullets}\n"
            "これらを再提示せず、別の有効な引用を選ぶか、in_body=false としてください。"
        )
    if failed_corrections:
        bullets = "\n".join(f"- 「{q}」" for q in failed_corrections)
        blocks.append(
            "前回の判定で correct_evidence_excerpt として以下を提示しましたが、"
            "いずれも <本文> 中に文字どおりには存在しませんでした:\n"
            f"{bullets}\n"
            "<本文> 内に実在する文字列をそのまま引用してください（要約・言い換え・"
            "改行をまたいだ連結は禁止）。"
        )
    feedback_block = ("\n\n" + "\n\n".join(blocks)) if blocks else ""
    return f"""人物が <本文> 中に実際に登場するかの判定。

前回の抽出で「{target_canonical_name}」が今回の登場人物として挙げられました。
そのとき evidence_excerpt として「{target_evidence_excerpt}」が示されましたが、
これは不適切と判定されました（<本文> 中に存在しない、対象を指していない、
あるいは捏造の可能性があります）。{feedback_block}

そこで改めて、「{target_canonical_name}」が今回の <本文> の中で
（どのような呼び方であれ）実際に言及されているかを判定してください。

ルール:
- 今回の <本文> のみを根拠とする。他の文脈は無視する。
- 「{target_canonical_name}」の名前そのもの・代名詞・関係呼び・役職呼び・代替の呼び名など、
  どの形であっても <本文> 中に言及があれば in_body=true。
- どの形でも <本文> 中に全く言及がなければ in_body=false。
- in_body=true の場合、その実際の言及箇所を correct_evidence_excerpt に書く。
  <本文> 内に文字どおりに存在し、かつ「{target_canonical_name}」を実際に指している部分を選ぶ
  （要約・言い換え・連結禁止、50文字以内目安）。
- in_body=false の場合、correct_evidence_excerpt は null。
- reason に判定の根拠を一文で。

<本文>
{episode_text}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "in_body": true または false,
    "correct_evidence_excerpt": "<本文> 内の実在する引用" または null,
    "reason": "判定の根拠"
}}
"""


VERIFY_QUOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "in_body": {"type": "boolean"},
        "correct_evidence_excerpt": {
            "oneOf": [
                {"type": "string", "maxLength": 200},
                {"type": "null"},
            ],
        },
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["in_body", "correct_evidence_excerpt", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------- merge_eval -


def char_merge_eval_prompt(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
) -> str:
    """Per-target likelihood-of-match evaluation against every registry entry,
    plus a virtual "新規" entry for "target is a new character not in registry".

    Layout: rules → summary → registry (backstory) → prev body → current
    names → 本文 → 対象 → registry (again, now that the model knows the
    target). The repeated registry gives the model a focused second read
    with the target in mind."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
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
- 既存の人物それぞれについて target, evidence, same_person_as_target を書く。
- target は判定対象のcanonical_nameをそのまま書く（毎エントリで同じ値）。
- 加えて「{NEW_CHAR_KEY}」キーで、対象が新規人物である可能性を同じ形式で評価する。
- evidence は本文・前回の本文・あらすじ・既存の人物情報を根拠にする。
- 推測ではなく、明示的な手がかり（呼び名・役割・関係・状況）を引用する。
- 「私」など語り手の一人称は、本文の語り手が誰なのかを慎重に考える。
{plot}{reg_block}{prev}{names_block}
<本文>
{episode_text}
</本文>

対象: 「{target_canonical_name}」
{reg_block}
JSONのみを出力すること。形式は以下のように。
{{
    "既存人物の canonical_name または「{NEW_CHAR_KEY}」": {{
        "target": "{target_canonical_name}",
        "evidence": "本文や既存情報からの具体的な手がかり",
        "same_person_as_target": "likely" | "possible" | "unlikely" | "impossible"
    }}
}}
"""


LIKELIHOOD_ENUM = ["likely", "possible", "unlikely", "impossible"]


def merge_eval_schema(existing_cnames: list[str], target_canonical_name: str):
    """{cname: {target, evidence, same_person_as_target}} for every existing
    cname plus the virtual NEW_CHAR_KEY entry. `target` is a const echo of
    the eval target (same anti-confusion trick as merge_judge_schema): the
    field name `same_person_as_target` only works if the model has the
    target identity in working state when it picks the likelihood."""
    item = {
        "type": "object",
        "properties": {
            "target": {"const": target_canonical_name},
            "evidence": {"type": "string", "maxLength": 800},
            "same_person_as_target": {"type": "string", "enum": LIKELIHOOD_ENUM},
        },
        "required": ["target", "evidence", "same_person_as_target"],
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
    episode_text: str,
    prev_episode_text: str,
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
    prev = prev_episode_text_block(prev_episode_text)
    cand_block = render_candidates(candidate_registry)
    target_block = f"\n<判定対象>\n{target_canonical_name}\n</判定対象>\n"
    return f"""人物の同一性判定。

「判定対象」が「候補人物」のそれぞれと同じ人物かを、候補ごとに真偽で判断してください。
「{NEW_CHAR_KEY}」は「候補人物のいずれでもない、新規の人物である」可能性を表します。

ルール:
- 候補人物それぞれについて target, evidence, same_person_as_target を書く。
- target は判定対象のcanonical_nameをそのまま書く（毎エントリで同じ値）。
- same_person_as_target は true（判定対象と同一人物だと思う）または false（別人だと思う）。
- evidence は本文・前回の本文・あらすじ・既存の人物情報を根拠にする。明示的な手がかり（呼び名・役割・関係・状況）を引用する。
- 判定対象は常に一人の特定の人物。same_person_as_target=true は候補人物の中で最も一致する一つ、または「{NEW_CHAR_KEY}」一つだけ。
  複数の候補を同時に true にしたり、既存候補と「{NEW_CHAR_KEY}」の両方を true にしたりしてはならない。
- 「{NEW_CHAR_KEY}」の same_person_as_target は「候補人物のいずれでもない新規の人物だ」と思うなら true。
- 「私」など語り手の一人称は、本文の語り手その人が誰なのかを慎重に考える。
{plot}{reg_block}{prev}{cand_block}{target_block}
<本文>
{episode_text}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "候補のcanonical_name または「{NEW_CHAR_KEY}」": {{
        "target": "{target_canonical_name}",
        "evidence": "判定の根拠",
        "same_person_as_target": true または false
    }}
}}
"""


def merge_judge_schema(candidate_cnames: list[str], target_canonical_name: str):
    """Per-candidate {target, evidence, same_person_as_target} for each
    candidate cname plus the virtual NEW_CHAR_KEY entry. `target` is a
    const echo of the judgement target — its only purpose is to keep the
    target identity hot in the model's working state when it emits the
    verdict (the per-target target gets diluted by long body text in
    weak-model 4B/8B runs). Downstream collapses to existing_ids (cnames
    with same_person_as_target=true, excl NEW_CHAR_KEY)."""
    item = {
        "type": "object",
        "properties": {
            "target": {"const": target_canonical_name},
            "evidence": {"type": "string", "maxLength": 500},
            "same_person_as_target": {"type": "boolean"},
        },
        "required": ["target", "evidence", "same_person_as_target"],
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


def _same_name_existing(registry: dict, target_canonical_name: str) -> list[dict]:
    """Existing registry entries whose canonical_name or aliases literally
    match the target. Used by char_merge_prune_prompt to flag suspected
    duplicates that judge slipped through."""
    hits = []
    for e in registry.values():
        names = {e.get("canonical_name"), *e.get("aliases", [])}
        if target_canonical_name in names:
            hits.append(e)
    return hits


def char_merge_prune_prompt(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    names: list[dict],
    target_canonical_name: str,
    target_description: str,
) -> str:
    """For a newly-extracted character (after merge said 'new'), decide if it
    is actually a character worth tracking, or extraction noise.

    Surfaces existing registry entries with matching canonical_name/aliases
    in a dedicated block. Judge has already said "not the same person" for
    these, but on weak models that verdict is often wrong; the per-name
    extract entry getting through to prune with the same name as an
    existing tracked character is overwhelmingly a duplicate. The prompt
    asks the model to default to drop unless it can clearly justify the
    coincidence."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
    names_block = render_names(names)
    same_name = _same_name_existing(registry, target_canonical_name)
    if same_name:
        slim = {e["canonical_name"]: _slim_entry(e) for e in same_name}
        same_name_block = (
            "\n<同名の既存人物>\n"
            + json.dumps(slim, ensure_ascii=False, indent=1)
            + "\n</同名の既存人物>\n"
        )
        same_name_rule = (
            "- <同名の既存人物> に判定対象と同じ呼び名の人物が登録されている。"
            "別人だと本文から明確に判別できなければ keep=false。"
            "（同一人物なら統合判定で既に扱われているため、ここに来ている時点で重複した抽出ミスの可能性が高い）\n"
        )
    else:
        same_name_block = ""
        same_name_rule = ""
    return f"""新規登録人物の妥当性判定。

判定対象は今回新しく抽出された人物候補です。これが追跡する価値のある具体的な人物か、
それとも既存の人物の別呼称・別名なのかを判断してください。

ルール:
- 本文に登場する具体的な個人（固有・特定可能な存在）で、<既存の人物> のいずれとも別人なら
  keep=true、merge_into=null。
- 一般化された群や役割の総称、本文内で語られる物語の中の登場人物、
  一過性の言及で追跡する価値がないものは keep=false、merge_into=null。
- 判定対象が <既存の人物> のうちある人物 X の別呼称・別名・本名・あだ名であり、
  X と同一人物だと判断できる場合は keep=false、merge_into に X の canonical_name を書く。
  （例: 本文で「タブロは父の名前」とあり、<既存の人物> に「父」が存在するなら merge_into="父"。
   この場合、判定対象は「父」のエイリアスとして登録される。）
{same_name_rule}- reason に短く根拠を述べる。
{plot}{reg_block}{prev}{names_block}{same_name_block}
<本文>
{episode_text}
</本文>

判定対象: 「{target_canonical_name}」
判定対象の説明: {target_description}

JSONのみを出力すること。形式は以下のように。
{{
    "reason": "判定の根拠",
    "merge_into": "既存の人物の canonical_name" または null,
    "keep": true または false
}}
"""


# --------------------------------------------------------------- merge_confirm


def char_merge_confirm_prompt(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    target_canonical_name: str,
    target_evidence_excerpt: str,
    proposed_merge_cname: str,
    proposed_merge_entry: dict,
) -> str:
    """Per-merge-proposal final-pushback check, invoked only when prune's
    merge_into is eval-tier unlikely/impossible — i.e. when prune
    contradicts eval. Phrased as a clean binary same-person question:
    target X vs candidate Y; body + registry as evidence.

    Note: prune's `reason` is deliberately NOT included here. When prune
    hallucinates an existing target (e.g. "既存の人物「村長」が存在する"
    when registry has no 村長) the grammar enum overrides the model's
    intended merge_into to a different existing name via prefix bias,
    leaving the reason text inconsistent with the actual merge_into.
    Showing that inconsistent reason to confirm makes the weak model
    copy it verbatim and propagate the hallucination."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
    target_block = (
        "\n<判定対象>\n"
        f"canonical_name: {target_canonical_name}\n"
        f"evidence_excerpt: {target_evidence_excerpt}\n"
        "</判定対象>\n"
    )
    proposed_block = (
        "\n<統合候補>\n"
        + json.dumps(
            {proposed_merge_cname: _slim_entry(proposed_merge_entry)},
            ensure_ascii=False,
            indent=1,
        )
        + "\n</統合候補>\n"
    )
    return f"""人物統合の最終確認。

判定対象（今回新しく抽出された人物）と、統合候補（既存の人物のうちの一人）が
同じ人物かを、本文と既存情報をもとに判定してください。

ルール:
- target は判定対象の canonical_name をそのまま書く。
- evidence は本文・前回の本文・あらすじ・既存の人物情報を根拠にする。
  明示的な手がかり（呼び名・役割・関係・状況）を引用する。
- same_person_as_target=true なら、判定対象は「統合候補」と同一人物であり、
  そのエイリアスとして統合される。
- same_person_as_target=false なら、判定対象は別人として新規登録される。
- 「私」など語り手の一人称は、本文の語り手その人が誰なのかを慎重に考える。
{plot}{reg_block}{prev}{proposed_block}{target_block}
<本文>
{episode_text}
</本文>

JSONのみを出力すること。形式は以下のように。
{{
    "target": "{target_canonical_name}",
    "evidence": "判定の根拠",
    "same_person_as_target": true または false
}}
"""


def merge_confirm_schema(target_canonical_name: str):
    """`target` is const-echoed (same anti-confusion trick as merge_judge).
    Field order: target → evidence → same_person_as_target so the model
    commits the boolean after writing the rationale."""
    return {
        "type": "object",
        "properties": {
            "target": {"const": target_canonical_name},
            "evidence": {"type": "string", "maxLength": 500},
            "same_person_as_target": {"type": "boolean"},
        },
        "required": ["target", "evidence", "same_person_as_target"],
        "additionalProperties": False,
    }


def merge_prune_schema(existing_cnames: list[str]):
    """Per-new-target keep/merge/drop verdict. `merge_into` is constrained
    to an existing canonical_name (or null) so the model can't invent
    targets. Field order is reason → merge_into → keep so the model
    commits to the decision after writing the rationale (chain-of-thought
    via field order)."""
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 500},
            "merge_into": {
                "oneOf": [
                    {"type": "string", "enum": list(existing_cnames)},
                    {"type": "null"},
                ],
            },
            "keep": {"type": "boolean"},
        },
        "required": ["reason", "merge_into", "keep"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------- consolidate


def char_consolidate_prompt(
    *,
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_entry: dict,
    target_surface_forms: list[str],
) -> str:
    """Per-target alias/canonical_name/description consolidation."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
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
{episode_text}
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
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
    target_id: str,
    target_entry: dict,
    new_aliases_this_episode: list[str] = (),
) -> str:
    """Per-target identity-change history entry. Only identity changes —
    not story events. Output may be null.

    `new_aliases_this_episode` lists aliases that were added to this
    entry by THIS episode's merge stage (not present in the pre-merge
    snapshot of the entry's aliases). A non-empty list strongly hints
    at an identity event (name reveal, role reveal) — without this
    signal the model sees the alias already in the entry and defaults
    to "no change". The list is surfaced in `<対象人物>` and an
    explicit rule tells the model to record the corresponding event."""
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
    target_block = (
        "\n<対象人物>\n"
        + json.dumps(
            {
                "id": target_id,
                "canonical_name": target_entry.get("canonical_name", ""),
                "aliases": target_entry.get("aliases", []),
                "new_aliases_this_episode": list(new_aliases_this_episode),
                "desc": target_entry.get("desc", ""),
                "history": target_entry.get("history", []),
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n</対象人物>\n"
    )
    new_alias_rule = (
        "- `new_aliases_this_episode` が空でない場合、今回のエピソードで対象人物に新たな呼び名が"
        "付いた・本名や別名が判明したことを意味する。<本文> を読んでその同一性イベントを"
        "特定し、new_history に記録すること（典型例: 本名判明、あだ名付与、役職・身分の判明）。\n"
        if new_aliases_this_episode
        else ""
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
{new_alias_rule}- reason に判定の根拠（変化の根拠 / なぜ無いのか）を短く書く。
{plot}{reg_block}{prev}{target_block}
<本文>
{episode_text}
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
    episode_text: str,
    prev_episode_text: str,
    prior_summaries: list[str],
    registry: dict,
) -> str:
    plot = summaries(prior_summaries)
    reg_block = render_registry(registry)
    prev = prev_episode_text_block(prev_episode_text)
    return f"""エピソードの要約。

今回のエピソード（第{seq}話）の出来事を一文か二文で要約してください。

ルール:
- 物語の進行に関わる主要な出来事のみ書く。
- 既知の人物・状況の繰り返しは避ける。
- 一文か二文（最大で二文）。100文字以内。
- 主観・感想・解説は書かない。事実のみ。
{plot}{reg_block}{prev}
<本文>
{episode_text}
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
