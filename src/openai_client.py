from __future__ import annotations

import os
import streamlit as st
from openai import OpenAI


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (env var or Streamlit secrets).")

    base_url = os.getenv("OPENAI_BASE_URL") or st.secrets.get("OPENAI_BASE_URL", None)
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)
