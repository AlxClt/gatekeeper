import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_api_token() -> str:
    """GATEKEEPER_API_TOKEN from the environment, falling back to the repo's .env file."""
    token = os.getenv("GATEKEEPER_API_TOKEN")
    if token:
        return token
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            key, _, value = line.partition("=")
            if key.strip() == "GATEKEEPER_API_TOKEN":
                return value.strip()
    return ""
