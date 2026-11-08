import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def ask_model(prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set OPENAI_API_KEY and OPENAI_MODEL before running.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Model API returned HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach model API: {error.reason}") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Model API returned an unexpected response.") from error

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Model API returned an unexpected response.") from error


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python 001_call_model_api.py "Your question"')
    print(ask_model(question))
