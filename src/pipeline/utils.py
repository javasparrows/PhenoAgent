import json
import re

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


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
