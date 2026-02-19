from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List


DATA_DIR = "data"
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")


def ensure_data_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_json_atomic(path: str, payload: Any) -> None:
    """
    Atomic-ish write: write to temp, then replace.
    """
    ensure_data_dirs()
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_session_files() -> List[str]:
    ensure_data_dirs()
    files: List[str] = []
    for fn in os.listdir(SESSIONS_DIR):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(SESSIONS_DIR, fn))
    files.sort(reverse=True)  # newest first by filename timestamp
    return files
