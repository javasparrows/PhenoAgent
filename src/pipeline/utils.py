import json
import re
import time
from dataclasses import dataclass

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

#: Per-call output cap of the submitted pipeline.
MAX_TOKENS = 512

#: Generation-prompt suffix of the harmony format (GPT-OSS).
HARMONY_ASSISTANT_SUFFIX = "<|start|>assistant"

#: Markers that open the final answer of a harmony response. The second form
#: appears when the detokenizer drops special tokens.
HARMONY_FINAL_MARKERS = ("<|channel|>final<|message|>", "assistantfinal")


def extract_json_from_response(response: str):
    """Extract and parse a JSON object from a raw LLM response.

    Prioritizes JSON wrapped in triple-backtick code blocks
    (```json ... ```).  Falls back to brace-matching heuristics and
    incomplete-JSON repair when the response is truncated.
    Returns {'error': ..., 'raw_response': ...} on failure.
    """
    # Try to find a fenced JSON code block
    pattern_code_block = r"```(?:json)?\s*(\{.*?\})\s*```"
    match_code_block = re.search(pattern_code_block, response, re.DOTALL)

    # Retry with a more permissive pattern that also accepts arrays
    if not match_code_block:
        pattern_code_block = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
        match_code_block = re.search(pattern_code_block, response, re.DOTALL)

    if match_code_block:
        json_str = match_code_block.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Fall through to next strategy

    # Handle truncated code blocks (opening ``` found but no closing ```)
    pattern_incomplete = r"```(?:json)?\s*(\{.*?)$"
    match_incomplete = re.search(pattern_incomplete, response, re.DOTALL)
    if match_incomplete:
        json_str = match_incomplete.group(1)
        json_str = try_fix_incomplete_json(json_str)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # No code block found -- extract the first { ... } span from the text
    json_start = response.find("{")
    if json_start != -1:
        brace_count = 0
        json_end = json_start
        for i, char in enumerate(response[json_start:], json_start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if brace_count == 0:  # Matching closing brace found
            json_str = response[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        else:  # Truncated -- no matching brace
            json_str = response[json_start:]
            json_str = try_fix_incomplete_json(json_str)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

    # Last resort: try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw_response": response}


def try_fix_incomplete_json(json_str: str) -> str:
    """Attempt to repair truncated JSON by closing open delimiters."""
    json_str = json_str.strip()

    # Close an unpaired double-quote
    quote_count = json_str.count('"') - json_str.count('\\"')
    if quote_count % 2 == 1:
        json_str += '"'

    # Remove a trailing comma
    if json_str.rstrip().endswith(","):
        json_str = json_str.rstrip().rstrip(",")

    # Close open braces
    open_braces = json_str.count("{") - json_str.count("}")
    if open_braces > 0:
        json_str += "}" * open_braces

    # Close open brackets
    open_brackets = json_str.count("[") - json_str.count("]")
    if open_brackets > 0:
        json_str += "]" * open_brackets

    return json_str


def load_model_and_tokenizer(model_path: str):
    """Load an MLX-LM model and its tokenizer."""
    model, tokenizer = load(model_path)
    return model, tokenizer


def run_inference(model, tokenizer, chat, temp=0.0, top_p=1.0) -> str:
    """Generate text from a chat-formatted prompt or a plain string.

    Args:
        model: MLX model instance.
        tokenizer: MLX tokenizer instance.
        chat: A list of chat-style message dicts, or a plain prompt string.
        temp: Sampling temperature.
        top_p: Nucleus sampling probability.

    Returns:
        The generated response string.
    """
    if isinstance(chat, list):
        # Convert chat-format list to a JSON string
        chat_str = json.dumps(chat, ensure_ascii=False, indent=2)
    else:
        chat_str = chat

    sampler = make_sampler(temp=temp, top_p=top_p)

    # Cap max_tokens to prevent runaway generation
    response = generate(
        model,
        tokenizer,
        prompt=chat_str,
        max_tokens=512,
        verbose=False,
        sampler=sampler,
    )
    return response


@dataclass(frozen=True)
class InferenceStats:
    """Runtime record of a single generation call."""

    seconds: float
    prompt_tokens: int
    gen_tokens: int
    finish_reason: str  # "eos" | "early_stop" | "length"
    chat_template: bool  # False if requested but the tokenizer has none


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold system turns into the first user turn for templates without a
    system role."""
    system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system_text:
        return rest
    if rest and rest[0]["role"] == "user":
        return [
            {"role": "user", "content": f"{system_text}\n\n{rest[0]['content']}"},
            *rest[1:],
        ]
    return [{"role": "user", "content": system_text}, *rest]


def build_chat_prompt(tokenizer, chat: list | str) -> str:
    """Render a prompt with the model's own chat template.

    A plain string becomes a single user turn. If the template rejects a
    system turn, or silently drops it, the system text is merged into the
    first user turn instead. GPT-OSS checkpoints ship a harmony-format
    template, which this renders like any other.
    """
    if isinstance(chat, str):
        messages = [{"role": "user", "content": chat}]
    else:
        messages = [dict(m) for m in chat]

    system_texts = [m["content"] for m in messages if m["role"] == "system"]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if all(text in rendered for text in system_texts):
            return rendered
    except Exception:
        # Jinja templates signal an unsupported role via raise_exception();
        # the concrete exception class differs between templates.
        if not system_texts:
            raise
    return tokenizer.apply_chat_template(
        _merge_system_into_user(messages), add_generation_prompt=True, tokenize=False
    )


def reasoning_mode(prompt_text: str) -> str:
    """Classify how a model separates reasoning from its answer.

    Returns "harmony" for GPT-OSS, "think" when the template opens a
    <think> block for the model, otherwise "none".
    """
    stripped = prompt_text.rstrip()
    if stripped.endswith(HARMONY_ASSISTANT_SUFFIX):
        return "harmony"
    if stripped.endswith("<think>"):
        return "think"
    return "none"


def strip_reasoning(text: str, mode: str) -> str | None:
    """Return the answer part of a response, or None while still reasoning."""
    if mode == "harmony":
        positions = [
            (text.rfind(marker), marker)
            for marker in HARMONY_FINAL_MARKERS
            if marker in text
        ]
        if not positions:
            return None
        pos, marker = max(positions)
        return text[pos + len(marker) :]
    if mode == "think" or "<think>" in text:
        if "</think>" not in text:
            return None
        return text[text.rfind("</think>") + len("</think>") :]
    return text


def _first_json_span(text: str, start: int) -> int | None:
    """End index (exclusive) of the bracketed span opening at text[start],
    or None if it has not closed yet."""
    closing = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in closing:
            stack.append(closing[char])
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return i + 1
    return None


def has_complete_json(text: str, mode: str = "none") -> bool:
    """Whether the answer already contains a complete, parseable JSON value.

    Only the first top-level bracketed span is considered, so a nested
    object never ends generation before its parent closes. A span that
    closes but does not parse (e.g. bracketed prose) is skipped.
    """
    answer = strip_reasoning(text, mode)
    if answer is None:
        return False
    search_from = 0
    while True:
        starts = [
            i
            for i in (answer.find("{", search_from), answer.find("[", search_from))
            if i != -1
        ]
        if not starts:
            return False
        start = min(starts)
        end = _first_json_span(answer, start)
        if end is None:
            return False
        try:
            json.loads(answer[start:end])
            return True
        except json.JSONDecodeError:
            search_from = end


def _encode_prompt(tokenizer, prompt_text: str) -> list[int]:
    """Tokenize like mlx_lm.generate, without doubling a templated BOS."""
    bos = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special_tokens)


def run_inference_with_stats(
    model,
    tokenizer,
    chat,
    *,
    use_chat_template: bool = True,
    early_stop: bool = True,
    temp: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, InferenceStats]:
    """Generate a response and record its runtime.

    With use_chat_template=False and early_stop=False this produces the
    same text as run_inference (the submitted pipeline). Otherwise the
    prompt is rendered with the model's chat template (falling back to the
    raw prompt if the tokenizer has none) and/or generation
    stops as soon as a complete JSON value follows any reasoning section.
    When a reasoning section (</think> or harmony analysis) has closed,
    only the answer part is returned.

    Returns:
        (response text, InferenceStats)
    """
    from mlx_lm.generate import stream_generate

    t0 = time.perf_counter()
    # Base models such as Preferred-MedLLM-Qwen-72B ship no chat template.
    templated = use_chat_template and bool(getattr(tokenizer, "chat_template", None))
    if templated:
        prompt_text = build_chat_prompt(tokenizer, chat)
    elif isinstance(chat, list):
        prompt_text = json.dumps(chat, ensure_ascii=False, indent=2)
    else:
        prompt_text = chat
    mode = reasoning_mode(prompt_text) if templated else "none"
    prompt_ids = _encode_prompt(tokenizer, prompt_text)

    sampler = make_sampler(temp=temp, top_p=top_p)
    text = ""
    finish_reason = "length"
    gen_tokens = 0
    stream = stream_generate(
        model, tokenizer, prompt=prompt_ids, max_tokens=max_tokens, sampler=sampler
    )
    try:
        for response in stream:
            text += response.text
            gen_tokens = response.generation_tokens
            if response.finish_reason is not None:
                finish_reason = "eos" if response.finish_reason == "stop" else "length"
                break
            if (
                early_stop
                and ("}" in response.text or "]" in response.text)
                and has_complete_json(text, mode)
            ):
                finish_reason = "early_stop"
                break
    finally:
        stream.close()
    seconds = time.perf_counter() - t0

    if templated:
        answer = strip_reasoning(text, mode)
        if answer is not None:
            text = answer
    stats = InferenceStats(
        seconds=seconds,
        prompt_tokens=len(prompt_ids),
        gen_tokens=gen_tokens,
        finish_reason=finish_reason,
        chat_template=templated,
    )
    return text, stats


def build_first_prompt(query: str) -> list:
    """Build the first-stage prompt (Semantic Parser) in chat format.

    Matches Supplementary Box 1 verbatim.
    """
    prompt_template = f"""以下はまず出力例を示しています。
心不全の患者の中で、肺水腫の患者は何人？"という質問に答えるために最低限必要な医学的な構造化項目を書いてください。json形式で出力してください。
回答：
[
  {{
    "fieldName": "心不全診断",
    "fieldType": "ブール値",
    "description": "患者が心不全と診断されているかどうか (true または false)",
  }},
  {{
    "fieldName": "肺水腫診断",
    "fieldType": "ブール値",
    "description": "患者が肺水腫と診断されているかどうか (true または false)"
  }}
]
出力例はここまでです。続いて、以下の質問に答えるために最低限必要な医学的な構造化項目を書いてください。jsonとしてparseできるようにjson形式で出力してください。回答のjsonのみを1回だけ出力してください。
回答：

# Query
{query}"""

    formatted_prompt = (
        "あなたは医療に関する質問に答えるAIアシスタントです。以下の質問文を構造化してください。\n"
        + prompt_template
    )

    chat = [
        {
            "role": "system",
            "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
        },
        {"role": "user", "content": formatted_prompt},
    ]
    return chat


def build_second_prompt(structured_question: str, document_text: str) -> list:
    """Build the second-stage prompt (Semantic Evaluator) in chat format.

    Matches Supplementary Box 2 verbatim.
    """
    prompt_template = (
        "json_for_structuring:\n"
        f"{structured_question}\n\n"
        "上記のjsonに従って、以下のデータを構造化してください。\n"
        f"ehr_data:\n{document_text}\n\n"
        "== json出力の例 ==\n"
        "```json\n"
        "{\n"
        '  "心停止診断": {\n'
        '    "value": false,\n'
        '    "reason": "そのvalueを選択した詳細な理由を書く"\n'
        "  },\n"
        '  "ECMO使用": {\n'
        '    "value": false,\n'
        '    "reason": "そのvalueを選択した詳細な理由を書く"\n'
        "  }\n"
        "}\n"
        "```\n"
        "== json出力の例 ==\n\n"
        "json_for_structuringのそれぞれについて、ehr_dataに該当するかどうかをtrueかfalseで出力してください。\n"
        "また、その理由も書いてください。\n"
        "回答は上の例のように、トップレベルが1つのJSONオブジェクトとなるように出力してください。\n"
        "回答："
    )

    formatted_prompt = (
        "あなたは医療に関する質問に答えるAIアシスタントです。以下の質問文を構造化してください。\n"
        + prompt_template
    )

    chat = [
        {
            "role": "system",
            "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
        },
        {"role": "user", "content": formatted_prompt},
    ]
    return chat


def remove_before_think(text):
    """Remove all text preceding the </think> token in chain-of-thought output."""
    match = re.search(r"</think>\n", text)
    if match:
        return text[match.end() :]  # Return everything after </think>
    return text  # Return unchanged if </think> is absent
