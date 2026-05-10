import json
import sys
from pathlib import Path

# 把 src 加入 Python 路径
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from openai import OpenAI
from forecasting_system.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
)

def main():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未读取到，请检查 .env 和 config.py")

    print("API KEY loaded (first 6 chars):", DEEPSEEK_API_KEY[:6])

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    resp = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reply ONLY in JSON."},
            {"role": "user", "content": 'Return {"status":"ok","message":"hello"} as JSON only.'},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    print("Raw response:", content)
    print("Parsed JSON:", json.loads(content))

if __name__ == "__main__":
    main()