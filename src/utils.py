from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List


def stable_hash(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def now_timestamp_id() -> str:
    # filesystem-friendly timestamp id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:180]


def difficulty_to_b(difficulty_label: str) -> float:
    m = {"easy": -1.0, "med": 0.0, "hard": 1.0}
    return float(m.get(difficulty_label, 0.0))


def sigmoid(x: float) -> float:
    # numerically safe enough for our ranges
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)
