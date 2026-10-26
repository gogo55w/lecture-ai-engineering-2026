"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問から会話を始める

引数を指定すると1回だけ質問し、指定しないと会話履歴を保って続けて質問できる。
ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] 入力トークン数: {usage.prompt_token_count}, "
            f"出力トークン数: {usage.candidates_token_count}"
        )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)
    question = DEFAULT_QUESTION

    while True:
        print(f"[質問] {question}\n")
        response = session.send_message(question)
        print_usage(response)
        print(response.text)

        while True:
            try:
                question = input("\n次の質問 (終了: exit)> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if question.lower() in {"exit"}:
                return
            if question:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.question:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
        else:
            chat()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
