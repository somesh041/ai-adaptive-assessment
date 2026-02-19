from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st

from src.adaptive_engine import AdaptiveSession, build_adaptive_session, select_next_item
from src.dashboard_utils import (
    compute_confidence_heuristic,
    make_difficulty_path_fig,
    make_mastery_heatmap_fig,
    make_theta_fig,
    summarize_misconceptions,
)
from src.itembank import generate_or_load_item_bank
from src.report_pdf import build_pdf_report_bytes
from src.schemas import Item, ItemBank
from src.storage import (
    ensure_data_dirs,
    list_session_files,
    load_json,
    save_json_atomic,
)
from src.utils import now_timestamp_id, safe_filename


APP_TITLE = "AI-Driven Adaptive Assessment & Diagnostic Intelligence System"
DEFAULT_MODEL = "gpt-4.1-mini"


def _get_openai_key_present() -> bool:
    # Presence check only; actual reading is handled in src/openai_client.py
    return bool(os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""))


def _init_state() -> None:
    if "item_bank_path" not in st.session_state:
        st.session_state.item_bank_path = None
    if "item_bank" not in st.session_state:
        st.session_state.item_bank = None
    if "adaptive" not in st.session_state:
        st.session_state.adaptive = None  # AdaptiveSession
    if "active_test_total" not in st.session_state:
        st.session_state.active_test_total = 15
    if "latest_session_path" not in st.session_state:
        st.session_state.latest_session_path = None
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL


def _load_item_bank_from_path(path: str) -> ItemBank:
    data = load_json(path)
    return ItemBank.model_validate(data)


def _render_generate_tab() -> None:
    st.subheader("1) Item Bank Generator")

    colA, colB = st.columns([2, 1])
    with colA:
        topic = st.text_input("Topic", value="Fractions & Decimals")
        grade = st.text_input("Grade", value="5")
        skills_csv = st.text_input(
            "Skills (comma separated)",
            value="place value, comparing decimals, fraction equivalence, converting fractions to decimals",
        )
    with colB:
        st.session_state.model = st.text_input("OpenAI model", value=st.session_state.model)
        n_items = st.number_input("Number of items", min_value=5, max_value=200, value=30, step=5)

    st.caption(
        "Generates MCQs in **strict JSON** and validates with Pydantic. "
        "If invalid, the system auto-repairs by calling the model again."
    )

    generate_btn = st.button("Generate / Load Item Bank", type="primary", use_container_width=True)

    if generate_btn:
        if not _get_openai_key_present():
            st.error("OpenAI API key not found. Set env var OPENAI_API_KEY or Streamlit secret OPENAI_API_KEY.")
            st.stop()

        skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
        if not skills:
            st.error("Please provide at least one skill.")
            st.stop()

        with st.spinner("Generating (or loading cached) item bank..."):
            bank, path = generate_or_load_item_bank(
                topic=topic.strip(),
                grade=grade.strip(),
                skills=skills,
                n_items=int(n_items),
                model=st.session_state.model,
            )

        st.session_state.item_bank = bank
        st.session_state.item_bank_path = path

        st.success(f"Item bank ready: {path}")
        st.write(f"Items: **{len(bank.items)}** | Unique skills: **{len(set(i.skill for i in bank.items))}**")

    # If already loaded, display preview + download
    if st.session_state.item_bank is not None and st.session_state.item_bank_path is not None:
        bank: ItemBank = st.session_state.item_bank
        path: str = st.session_state.item_bank_path

        st.markdown("#### Preview (first 5)")
        st.json([i.model_dump() for i in bank.items[:5]])

        st.markdown("#### Download")
        with open(path, "rb") as f:
            st.download_button(
                "Download item bank JSON",
                data=f.read(),
                file_name=os.path.basename(path),
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("#### Load a different item bank")
    else:
        st.markdown("#### Load an existing item bank")

    uploaded = st.file_uploader("Upload item bank JSON", type=["json"])
    if uploaded is not None:
        try:
            data = json.loads(uploaded.getvalue().decode("utf-8"))
            bank = ItemBank.model_validate(data)
            # Persist uploaded to data/ for reuse
            ensure_data_dirs()
            filename = safe_filename(uploaded.name) or f"itembank_uploaded_{now_timestamp_id()}.json"
            save_path = os.path.join("data", filename)
            save_json_atomic(save_path, bank.model_dump(mode="json"))
            st.session_state.item_bank = bank
            st.session_state.item_bank_path = save_path
            st.success(f"Loaded and saved: {save_path}")
        except Exception as e:
            st.error(f"Invalid item bank JSON: {e}")


def _render_test_tab() -> None:
    st.subheader("2) Adaptive Test Engine (No LLM calls during quiz)")

    if st.session_state.item_bank is None:
        st.info("Generate or load an item bank first (Generate tab).")
        return

    bank: ItemBank = st.session_state.item_bank

    st.session_state.active_test_total = int(
        st.slider("Number of questions in this test", min_value=5, max_value=40, value=15, step=1)
    )

    session_name = st.text_input("Session name (optional)", value="")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start new test", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("Reset current test", use_container_width=True)

    if reset_btn:
        st.session_state.adaptive = None
        st.toast("Test reset.")

    if start_btn or st.session_state.adaptive is None:
        st.session_state.adaptive = build_adaptive_session(
            item_bank=bank,
            total_questions=st.session_state.active_test_total,
            session_name=session_name.strip() or None,
        )

    sess: AdaptiveSession = st.session_state.adaptive

    # If finished, persist and guide to dashboard
    if sess.is_finished:
        ensure_data_dirs()
        ts_id = sess.session_id
        out_path = os.path.join("data", "sessions", f"{ts_id}.json")
        payload = sess.to_json_dict()
        save_json_atomic(out_path, payload)
        st.session_state.latest_session_path = out_path

        st.success("Test completed and saved.")
        st.code(out_path)

        st.info("Go to **Dashboard** tab to view diagnostics and export a PDF report.")
        return

    # Select next item (deterministic based on current state; no LLM)
    next_item = select_next_item(sess, bank)

    st.markdown(f"**Progress:** {len(sess.answers)} / {sess.total_questions}")
    st.progress(len(sess.answers) / sess.total_questions)

    st.markdown("---")
    st.markdown(f"### Q{len(sess.answers) + 1}")
    st.markdown(f"**Skill:** {next_item.skill}  \n**Difficulty:** {next_item.difficulty_label}  \n**Bloom:** {next_item.bloom_level}")
    st.write(next_item.stem)

    choice = st.radio(
        "Choose an option:",
        options=list(range(4)),
        format_func=lambda idx: f"{chr(65 + idx)}. {next_item.options[idx]}",
        key=f"choice_{sess.step_index}",
    )

    submitted = st.button("Submit answer", type="primary", use_container_width=True)

    if submitted:
        was_correct = int(choice) == int(next_item.correct_index)
        sess.apply_answer(item=next_item, chosen_index=int(choice), was_correct=was_correct)
        st.toast("Correct ✅" if was_correct else "Incorrect ❌")

        # Persist incremental session snapshots too (crash-safe)
        ensure_data_dirs()
        out_path = os.path.join("data", "sessions", f"{sess.session_id}.json")
        save_json_atomic(out_path, sess.to_json_dict())
        st.session_state.latest_session_path = out_path

        # Immediate feedback (short)
        with st.expander("Explanation", expanded=not was_correct):
            st.write(next_item.explanation_short)

        st.rerun()

    # Quick live stats
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Theta", f"{sess.theta:.2f}")
    with c2:
        st.metric("Answered", len(sess.answers))
    with c3:
        st.metric("Accuracy", f"{(sess.correct_count / max(1, len(sess.answers)) * 100):.0f}%")

    if st.checkbox("Show mastery (live)", value=False):
        st.json(sess.mastery)


def _render_dashboard_tab() -> None:
    st.subheader("3) Diagnostic Dashboard")

    ensure_data_dirs()
    session_files = list_session_files()

    default_path = st.session_state.latest_session_path if st.session_state.latest_session_path in session_files else None

    if not session_files:
        st.info("No saved sessions yet. Complete a test in the Test tab.")
        return

    selected = st.selectbox(
        "Choose a session",
        options=session_files,
        index=session_files.index(default_path) if default_path else 0,
        format_func=lambda p: os.path.basename(p),
    )

    sess_data = load_json(selected)
    sess = AdaptiveSession.from_json_dict(sess_data)

    # Core metrics
    confidence = compute_confidence_heuristic(sess)
    st.markdown("#### Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Theta", f"{sess.theta:.2f}")
    with c2:
        st.metric("Answered", len(sess.answers))
    with c3:
        st.metric("Accuracy", f"{(sess.correct_count / max(1, len(sess.answers)) * 100):.0f}%")
    with c4:
        st.metric("Confidence (heuristic)", f"{confidence*100:.0f}%")

    # Charts
    st.markdown("#### Charts")
    fig_theta = make_theta_fig(sess)
    st.pyplot(fig_theta, clear_figure=True)

    fig_heat = make_mastery_heatmap_fig(sess)
    st.pyplot(fig_heat, clear_figure=True)

    fig_diff = make_difficulty_path_fig(sess)
    st.pyplot(fig_diff, clear_figure=True)

    # Misconceptions
    st.markdown("#### Misconceptions")
    top_mis = summarize_misconceptions(sess, top_k=8)
    if not top_mis:
        st.info("No misconception signals yet (no distractor misconceptions recorded).")
    else:
        st.table([{"misconception": k, "count": v} for k, v in top_mis])

    st.markdown("---")
    st.subheader("AI Narrative & Recommendations (1 LLM call)")

    if not _get_openai_key_present():
        st.warning("OpenAI API key not found. Set env var OPENAI_API_KEY or Streamlit secret OPENAI_API_KEY.")
        return

    from src.dashboard_utils import generate_ai_insights_one_call  # local import to avoid any accidental calls

    if sess.ai_insights is None:
        st.caption("Generates ≤160-word narrative + exactly 5 bullet recommendations.")
        if st.button("Generate AI insights", type="primary", use_container_width=True):
            with st.spinner("Generating AI insights..."):
                ai = generate_ai_insights_one_call(
                    sess=sess,
                    model=st.session_state.model,
                    top_misconceptions=top_mis,
                    confidence=confidence,
                )
                sess.ai_insights = ai
                # Persist back into session file
                save_json_atomic(selected, sess.to_json_dict())
                st.rerun()
        else:
            st.info("Click to generate AI insights for this session.")
            return

    # Display AI insights
    st.markdown("#### Narrative")
    st.write(sess.ai_insights["narrative"])

    st.markdown("#### Recommendations")
    for b in sess.ai_insights["recommendations"]:
        st.markdown(f"- {b}")

    st.markdown("---")
    st.subheader("4) Export PDF Report")

    # Build figures again for report embedding (fresh fig objects)
    figs = {
        "theta": make_theta_fig(sess),
        "mastery": make_mastery_heatmap_fig(sess),
        "difficulty": make_difficulty_path_fig(sess),
    }

    report_name = f"adaptive_report_{os.path.basename(selected).replace('.json','')}.pdf"

    if st.button("Generate PDF", type="primary", use_container_width=True):
        with st.spinner("Building PDF report..."):
            pdf_bytes = build_pdf_report_bytes(
                sess=sess,
                confidence=confidence,
                top_misconceptions=top_mis,
                figures=figs,
                title=APP_TITLE,
            )
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=report_name,
            mime="application/pdf",
            use_container_width=True,
        )

    # Cleanup figs to avoid memory buildup
    for f in figs.values():
        plt.close(f)


def main() -> None:
    ensure_data_dirs()
    _init_state()

    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.caption("Streamlit demo repo: item generation + adaptive testing + diagnostics + PDF export.")

    # Top bar info
    with st.expander("System status", expanded=False):
        st.write(
            {
                "openai_key_present": _get_openai_key_present(),
                "model_default": DEFAULT_MODEL,
                "current_model": st.session_state.model,
                "item_bank_path": st.session_state.item_bank_path,
                "latest_session_path": st.session_state.latest_session_path,
            }
        )

    tab_generate, tab_test, tab_dash = st.tabs(["Generate", "Test", "Dashboard"])

    with tab_generate:
        _render_generate_tab()
    with tab_test:
        _render_test_tab()
    with tab_dash:
        _render_dashboard_tab()


if __name__ == "__main__":
    main()
