import os
import requests


def simple_call(prompt: str) -> str:
    """
    Execute a minimal chat completion request and return the content.
    """
    base_url = os.getenv("BASE_URL_LIMIT")
    api_key = os.getenv("API_KEY_LIMIT")

    if not base_url or not api_key:
        raise RuntimeError("Please set BASE_URL_LIMIT and API_KEY_LIMIT environment variables.")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": "gemini-2.5-pro-search",
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty response received: {data}")
    return choices[0]["message"]["content"]


if __name__ == "__main__":
    user_prompt = "简单介绍一下心理学中的自我效能感。"
    result = simple_call(user_prompt)
    print("模型回复:")
    print(result)
