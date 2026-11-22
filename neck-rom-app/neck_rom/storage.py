"""Save / load measurement sessions as JSON files: data/<user>/<timestamp>.json"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save_session(user: str, session: dict, source: str = "webcam") -> Path:
    folder = DATA_DIR / user
    folder.mkdir(parents=True, exist_ok=True)
    session = dict(session, user=user, source=source)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{stamp}.json"
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_sessions(user: str, limit: int = 12) -> list[dict]:
    folder = DATA_DIR / user
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.glob("*.json"))[-limit:]:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
