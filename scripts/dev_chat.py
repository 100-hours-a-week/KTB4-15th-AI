"""터미널에서 직접 대화해보는 개발용 클라이언트.

    python scripts/dev_chat.py

명령:
    /wishlist   찜 추천 칩을 누른 것처럼 보낸다 (대화 첫 턴에서만 된다)
    /new        새 채팅방으로 바꾼다
    /quit       끝낸다
"""

import json
import os
import random
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE = os.environ.get("AI_SERVER", "http://127.0.0.1:8000")
KEY = os.environ.get("INTERNAL_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}

GREY = "\033[90m"
CYAN = "\033[36m"
RESET = "\033[0m"


def send(chat_id: int, message: str, source_type: str = "GENERAL") -> None:
    body = {
        "chat_id": chat_id,
        "user_id": 10,
        "message": message,
        "source_type": source_type,
        "product_ids": [f"{i:07d}" for i in range(1, 11)] if source_type == "WISHLIST" else [],
    }

    with httpx.Client(timeout=120) as client, client.stream(
        "POST", f"{BASE}/api/v1/chat/stream", json=body, headers=HEADERS
    ) as response:
        if response.headers.get("content-type", "").startswith("application/json"):
            response.read()
            print(f"{GREY}[HTTP {response.status_code}] {response.text}{RESET}")
            return

        event = None
        printed_prefix = False
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                if event == "token":
                    if not printed_prefix:
                        print("봇: ", end="")
                        printed_prefix = True
                    print(data["content"], end="", flush=True)
                elif event == "status":
                    print(f"{GREY}… {data['label']}{RESET}")
                elif event == "products":
                    print()
                    items = data["products"]
                    if not items:
                        print(f"{GREY}(추천 상품 없음){RESET}")
                    for index, product in enumerate(items, start=1):
                        print(f"{CYAN}  {index}. {product['product_name']}{RESET}")
                        print(f"{GREY}     {product['color']} · {product['detail_url']}{RESET}")
                        if product["llm_comment"]:
                            print(f"{GREY}     {product['llm_comment']}{RESET}")
                elif event == "error":
                    print(f"\n{GREY}[error] {data['code']} — {data['message']}{RESET}")
                elif event == "done":
                    if printed_prefix:
                        print()
                    if data.get("content"):
                        print(f"{GREY}[done] 저장될 문장 {len(data['content'])}자{RESET}")


def main() -> None:
    chat_id = random.randint(1000, 9999)
    print(f"{GREY}{BASE} · chat_id={chat_id} · /wishlist /new /quit{RESET}")

    while True:
        try:
            text = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not text:
            continue
        if text == "/quit":
            return
        if text == "/new":
            chat_id = random.randint(1000, 9999)
            print(f"{GREY}새 채팅방 chat_id={chat_id}{RESET}")
            continue
        if text == "/wishlist":
            send(chat_id, "내 찜 목록으로 추천받기", source_type="WISHLIST")
            continue

        send(chat_id, text)


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("서버에 연결하지 못했습니다. 먼저 python scripts/dev_server.py 를 띄우세요.")
        sys.exit(1)
