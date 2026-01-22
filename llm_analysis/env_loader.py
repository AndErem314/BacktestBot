from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


@dataclass
class LLMConfig:
    provider: str  # 'openai' | 'gemini' | 'auto'
    openai_api_key: Optional[str]
    openai_model: Optional[str]
    gemini_api_key: Optional[str]
    gemini_model: Optional[str]
    temperature: float = 0.2


def _load_env_near_project_root():
    """Load .env from project root if python-dotenv is available.

    Robustly attempts common locations regardless of the current working directory.
    """
    if not _HAS_DOTENV:
        return
    here = Path(__file__).resolve()
    # Candidates: current CWD, project root (BacktestBot), and its parent as a fallback
    candidates = [
        Path.cwd() / '.env',
        here.parents[1] / '.env',   # .../BacktestBot/.env
        here.parents[2] / '.env',   # .../Python/.env (fallback)
    ]
    for p in candidates:
        if p.exists():
            # Do not override already-set environment variables
            load_dotenv(dotenv_path=p, override=False)
            break


def load_llm_config() -> LLMConfig:
    """Load LLM configuration from environment variables.

    Expected variables:
      - LLM_PROVIDER (optional): 'openai' | 'gemini' | 'auto'
      - OPENAI_API_KEY, OPENAI_MODEL
      - GEMINI_API_KEY, GEMINI_MODEL
      - LLM_TEMPERATURE (optional, default 0.2)
    """
    _load_env_near_project_root()

    provider = os.getenv('LLM_PROVIDER', 'auto').lower()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
    try:
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.2'))
    except ValueError:
        temperature = 0.2

    # Auto detect if both are present
    if provider == 'auto':
        if gemini_api_key and not openai_api_key:
            provider = 'gemini'
        elif openai_api_key and not gemini_api_key:
            provider = 'openai'
        elif openai_api_key and gemini_api_key:
            # Prefer OpenAI by default; can be overridden in call
            provider = 'openai'
        else:
            provider = 'openai'  # default, may error on missing key later

    return LLMConfig(
        provider=provider,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        temperature=temperature,
    )
