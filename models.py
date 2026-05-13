"""
ai_provider.py
--------------
Tries free-tier AI providers in order. If one hits a rate limit or quota,
it automatically falls through to the next one.

SETUP
-----
1. Install dependencies:
   pip install google-generativeai groq mistralai openai requests

2. Add keys to config.py (leave as None if you don't have that provider):
   GEMINI_API_KEY     = "AIza..."      # aistudio.google.com   — FREE, no card
   GROQ_API_KEY       = "gsk_..."      # console.groq.com      — FREE, no card
   MISTRAL_API_KEY    = "..."          # console.mistral.ai    — FREE, no card
   DEEPSEEK_API_KEY   = "sk-..."       # platform.deepseek.com — FREE 5M tokens
   OPENROUTER_API_KEY = "sk-or-..."    # openrouter.ai         — FREE tier

   # Optional paid fallbacks (only used if all free ones fail)
   OPENAI_API_KEY     = "sk-..."       # platform.openai.com
   ANTHROPIC_API_KEY  = "sk-ant-..."   # console.anthropic.com
"""

import importlib

# ── Load config keys safely ────────────────────────────────────────────────────
def _cfg(key):
    try:
        mod = importlib.import_module("config")
        return getattr(mod, key, None)
    except ModuleNotFoundError:
        return None

# ── Errors that mean "quota/rate limit → try next provider" ───────────────────
QUOTA_SIGNALS = (
    "429", "rate_limit", "rate limit", "quota", "insufficient_quota",
    "overloaded", "529", "resource_exhausted", "too many requests",
    "tokens per day", "requests per day", "exceeded"
)

def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in QUOTA_SIGNALS)


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# 1. Google Gemini — best free tier (1,500 req/day, no card, no expiry)
def _gemini(prompt: str) -> str:
    import google.generativeai as genai
    key = _cfg("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY in config.py")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# 2. Groq — very fast, ~1,000 req/day free (no card)
def _groq(prompt: str) -> str:
    from groq import Groq
    key = _cfg("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No GROQ_API_KEY in config.py")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 3. Mistral — free rate-limited tier (no card)
def _mistral(prompt: str) -> str:
    from mistralai import Mistral
    key = _cfg("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("No MISTRAL_API_KEY in config.py")
    client = Mistral(api_key=key)
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 4. DeepSeek — 5M free tokens on signup, OpenAI-compatible
def _deepseek(prompt: str) -> str:
    from openai import OpenAI
    key = _cfg("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("No DEEPSEEK_API_KEY in config.py")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 5. OpenRouter — 50+ models, free rate-limited access (no card)
def _openrouter(prompt: str) -> str:
    import requests
    key = _cfg("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("No OPENROUTER_API_KEY in config.py")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",  # free model tag
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# 6. OpenAI — paid fallback ($5 free credits for new accounts)
def _openai(prompt: str) -> str:
    from openai import OpenAI
    key = _cfg("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OPENAI_API_KEY in config.py")
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 7. Anthropic Claude — paid fallback ($5 free credits for new accounts)
def _anthropic(prompt: str) -> str:
    import anthropic
    key = _cfg("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY in config.py")
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER REGISTRY — edit order to change priority
# ══════════════════════════════════════════════════════════════════════════════
PROVIDERS = [
    ("Gemini",     _gemini),      # 1st — best free tier
    ("Groq",       _groq),        # 2nd — fastest
    ("Mistral",    _mistral),     # 3rd — free, no card
    ("DeepSeek",   _deepseek),    # 4th — generous free tokens
    ("OpenRouter", _openrouter),  # 5th — many free models
    ("OpenAI",     _openai),      # 6th — paid fallback
    ("Anthropic",  _anthropic),   # 7th — paid fallback
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def generate(prompt: str) -> str:
    """
    Try each provider in order. Skip providers with missing keys.
    Fall through on quota/rate-limit errors. Raise if all fail.
    """
    last_error = None

    for name, fn in PROVIDERS:
        try:
            result = fn(prompt)
            print(f"[ai_provider] ✅ Used: {name}")
            return result

        except RuntimeError as e:
            # Missing API key — silently skip this provider
            if "No " in str(e) and "in config.py" in str(e):
                continue
            raise

        except Exception as e:
            if _is_quota_error(e):
                print(f"[ai_provider] ⚠️  {name} quota/rate limit hit → trying next...")
                last_error = e
                continue
            # Unexpected error — re-raise so you can debug it
            raise

    raise RuntimeError(
        f"All AI providers failed or have no API keys configured.\n"
        f"Last quota error: {last_error}\n"
        f"Check that at least one API key is set in config.py."
    )