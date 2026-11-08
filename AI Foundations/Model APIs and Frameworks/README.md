# Model APIs and Frameworks

Practice integrating existing AI models into applications:

- Call hosted model APIs or use a local model framework.
- Handle prompts, structured outputs, timeouts, retries, and rate limits.
- Evaluate responses for quality and reliability.
- Protect credentials and avoid sending private data unintentionally.

`001_call_model_api.py` demonstrates an OpenAI-compatible chat-completions
request using only Python's standard library. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` in your environment; optionally set `OPENAI_BASE_URL` for a
compatible provider. Never put API keys in source control.
