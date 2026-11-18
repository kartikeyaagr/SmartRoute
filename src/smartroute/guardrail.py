"""
Input guardrail: cheap, deterministic checks run before any model is called.

Runs in the server (the trust boundary), not in the Router library — a library
caller already controls its own prompts. Three checks, in cost order:

  too_long          prompt over GUARDRAIL_MAX_CHARS. Caps the spend one request can
                    cause (a 1M-char prompt to the frontier tier is a cost-DoS).
  secret            the prompt contains something shaped like a credential. Sending
                    it on would leak it to a third-party provider and its logs.
  prompt_injection  well-known override phrasing. A denylist only catches the lazy
                    attacks; it is a tripwire, not a defence.

ponytail: regex denylist. Upgrade path is a classifier model (Llama Guard /
Prompt Guard) served on the self-hosted vLLM route; run GUARDRAIL=log first to
measure false positives before enforcing anything stricter.
"""

import re

_SECRET = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{20,}"          # Anthropic
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"   # OpenAI
    r"|gsk_[A-Za-z0-9]{20,}"              # Groq
    r"|AKIA[0-9A-Z]{16}"                  # AWS access key id
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)

_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above) instructions"
    r"|disregard (?:all |the |your )?(?:previous|prior|system) (?:instructions|prompt)"
    r"|(?:reveal|print|show) (?:me )?(?:your|the) system prompt",
    re.IGNORECASE,
)


def check_input(messages: list[dict], max_chars: int) -> str | None:
    """Return the reason a request should be blocked, or None if it passes."""
    # str() also flattens multimodal content parts ([{"type": "text", ...}]).
    text = "\n".join(str(m.get("content") or "") for m in messages)
    if len(text) > max_chars:
        return "too_long"
    if _SECRET.search(text):
        return "secret"
    if _INJECTION.search(text):
        return "prompt_injection"
    return None
