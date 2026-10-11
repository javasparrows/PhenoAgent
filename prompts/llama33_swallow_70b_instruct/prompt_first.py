def build_first_prompt(query: str) -> str:
    """Build the first-stage prompt (Semantic Parser) as a plain string.

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
    return formatted_prompt
