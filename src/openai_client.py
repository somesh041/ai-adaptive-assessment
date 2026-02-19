from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from openai import OpenAI


def get_openai_api_key() -> str:
    """
    Reads API key from env var OPENAI_API_KEY, and also supports Streamlit secrets.
    Priority: Streamlit secrets first (useful on Streamlit Cloud), then env var.
    """
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore[attr-defined]
    except Exception:
        # Not running in Streamlit or secrets not configured
        key = ""

    if not key:
        key = os.getenv("OPENAI_API_KEY", "") or ""

    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets or environment variables.")

    return key


_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_openai_api_key())
    return _client
