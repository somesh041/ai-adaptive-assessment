from __future__ import annotations

import glob
import json
import os
import tempfile
from typing import Any, Dict, List


def ensure_data_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "sessions"), exist_ok=True)


def save_json_atomic(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_session_files() -> List[str]:
    ensure_data_dirs()
    paths = glob.glob(os.path.join("data", "sessions", "*.json"))
    # newest first
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths
