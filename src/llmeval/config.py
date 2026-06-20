from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "support_intents.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

LABELS = ["billing", "technical", "account", "cancellation", "other"]
CHAT_MODEL = "claude-haiku-4-5-20251001"
