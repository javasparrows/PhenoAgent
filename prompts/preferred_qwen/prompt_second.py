def build_second_prompt(structured_question: str, document_text: str) -> str:
    """Build the second-stage prompt (Semantic Evaluator) as a plain string.

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
    return formatted_prompt
