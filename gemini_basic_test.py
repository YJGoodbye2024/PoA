import json
import os
import sys
import time
from itertools import cycle

import requests


def gather_credentials():
    creds = []
    base_default = os.getenv("BASE_URL_LIMIT")
    for idx in range(10):
        suffix = "" if idx == 0 else str(idx)
        api_key = os.getenv(f"API_KEY_LIMIT{suffix}")
        if not api_key:
            continue
        base_env = os.getenv(f"BASE_URL_LIMIT{suffix}") or base_default
        if not base_env:
            continue
        creds.append((base_env.rstrip("/"), api_key, f"gemini-{idx + 1}"))
    if not creds:
        raise RuntimeError("No Gemini credentials found.")
    return creds


def call_gemini(base_url, api_key, system_prompt, user_prompt, model="gemini-2.5-pro"):
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"HTTP {response.status_code}")
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Non-JSON response:", response.text[:200])
        return None
    print("Raw response:", json.dumps(data, ensure_ascii=False))
    choices = data.get("choices")
    if choices:
        content = choices[0].get("message", {}).get("content")
        if content:
            print("Content:", content)
            return content
    print("No content returned.")
    return None


def main():
    system_prompt = "你是一个只回答是或否的助手。"
    user_prompt = "天空是蓝色的吗？请只回答是或否。"

    creds = gather_credentials()
    cred_cycle = cycle(creds)

    for idx in range(5):
        base_url, api_key, name = next(cred_cycle)
        print(f"\n=== Attempt {idx + 1} using {name} ===")
        try:
            call_gemini(base_url, api_key, system_prompt, user_prompt)
        except Exception as exc:
            print(f"Error calling Gemini: {exc}")
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
