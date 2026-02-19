from __future__ import annotations

import io
import json
import os
import zipfile
from typing import List, Optional

import matplotlib.pyplot as plt
import streamlit as st

from src.adaptive_engine import AdaptiveSession, build_adaptive_session, select_next_item
from src.dashboard_utils import (
    compute_confidence_heuristic,
    generate_ai_insights_one_call,
    make_bank_info_curve_fig,
    make_difficulty_path_fig,
    make_mastery_heatmap_fig,
    make_theta_fig,
    summarize_misconceptions,
)
from src.itembank import generate_or_load_item_bank
from src.psychometrics.bank_stats import bank_summary_metrics
from src.psychometrics.calibration import calibrate_2pl_jml_from_sessions
from src.report_pdf import build_pdf_report_bytes
from src.schemas import ItemBank, LangCode
from src.storage import ensure_data_dirs, list_session_files, load_json, save_json_atomic
from src.translate import LANG_LABELS
from src.utils import now_timestamp_id, safe_filename


APP_TITLE = "AI-Driven Adaptive Assessment & Diagnostic Intelligence System"
DEFAULT_MODEL = "gpt-4.1-mini"


def _get_openai_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""))


def _init_state() -> None:
    st.session_state.setdefault("item_bank_path", None)
    st.session_state.setdefault("item_bank", None)
    st.session_state.setdefault("adaptive", None)
    st.session_state.setdefault("active_test_total", 15)
    st.session_state.setdefault("latest_session_path", None)
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("quiz_lang", "en")
    st.session_state.setdefault("generate_langs", [])
    st.session_state.setdefault("qc_enabled", True)


def _zip_files(paths: List[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=os.path.join("sessions", os.path.basename(p)))
    buf.seek(0)
    return buf.read()


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

        st.markdown("##### Robustness")
        st.session_state.qc_enabled = st.checkbox(
            "Enable Psychometric QC (review + auto-regenerate bad items)",
            value=st.session_state.qc_enabled,
            help="Adds extra model calls. Turn off for cheapest runs.",
        )

        st.markdown("##### Optional translations (OFF by default)")
        enable_i18n = st.checkbox("Generate translations", value=False)
        langs_selected: List[LangCode] = []
        if enable_i18n:
            langs_selected = st.multiselect(
                "Select Indian languages",
                options=list(LANG_LABELS.keys()),
                default=[],
                format_func=lambda c: LANG_LABELS[c],
            )
        st.session_state.generate_langs = langs_selected

    with colB:
        st.session_state.model = st.text_input("OpenAI model", value=st.session_state.model)
        n_items = st.number_input("Number of items", min_value=5, max_value=200, value=30, step=5)

    if st.button("Generate / Load Item Bank", type="primary", use_container_width=True):
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
                translate_langs=st.session_state.generate_langs or None,
                qc_enabled=st.session_state.qc_enabled,
            )

        st.session_state.item_bank = bank
        st.session_state.item_bank_path = path
        st.success(f"Item bank ready: {path}")

    if st.session_state.item_bank is None:
        st.info("Generate or upload an item bank to proceed.")
        uploaded = st.file_uploader("Upload item bank JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                bank = ItemBank.model_validate(data)
                ensure_data_dirs()
                filename = safe_filename(uploaded.name) or f"itembank_uploaded_{now_timestamp_id()}.json"
                save_path = os.path.join("data", filename)
                save_json_atomic(save_path, bank.model_dump(mode="json"))
                st.session_state.item_bank = bank
                st.session_state.item_bank_path = save_path
                st.success(f"Loaded and saved: {save_path}")
            except Exception as e:
                st.error(f"Invalid item bank JSON: {e}")
        return

    bank: ItemBank = st.session_state.item_bank
    st.markdown("#### Bank summary")
    st.json(bank_summary_metrics(bank))

    st.markdown("#### Preview (first 2 items)")
    st.json([i.model_dump() for i in bank.items[:2]])

    path = st.session_state.item_bank_path
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(
                "Download item bank JSON",
                data=f.read(),
                file_name=os.path.basename(path),
                mime="application/json",
                use_container_width=True,
            )


def _render_test_tab() -> None:
    st.subheader("2) Adaptive Test (NO LLM calls during quiz)")

    if st.session_state.item_bank is None:
        st.info("Generate or load an item bank first (Generate tab).")
        return

    bank: ItemBank = st.session_state.item_bank

    st.session_state.active_test_total = int(
        st.slider("Number of questions in this test", min_value=5, max_value=40, value=5, step=1)
    )

    available_langs: List[str] = ["en"]
    for lc in LANG_LABELS.keys():
        if any((it.translations or {}).get(lc) for it in bank.items):
            available_langs.append(lc)

    st.session_state.quiz_lang = st.selectbox(
        "Quiz display language",
        options=available_langs,
        index=0,
        format_func=lambda c: "English" if c == "en" else LANG_LABELS[c],  # type: ignore[index]
    )

    session_name = st.text_input("Session name (optional)", value="")

    col1, col2 = st.columns(2)
    start_btn = col1.button("Start new test", type="primary", use_container_width=True)
    reset_btn = col2.button("Reset current test", use_container_width=True)

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

    # finished: save + download (Streamlit Cloud storage is ephemeral)
    if sess.is_finished:
        ensure_data_dirs()
        out_path = os.path.join("data", "sessions", f"{sess.session_id}.json")
        save_json_atomic(out_path, sess.to_json_dict())
        st.session_state.latest_session_path = out_path

        st.success("Test completed and saved (runtime storage).")
        st.caption("Streamlit Cloud won’t push these files to GitHub automatically; download them below.")

        with open(out_path, "rb") as f:
            st.download_button(
                "Download this session JSON",
                data=f.read(),
                file_name=os.path.basename(out_path),
                mime="application/json",
                use_container_width=True,
            )
        return

    next_item = select_next_item(sess, bank)

    lang = st.session_state.quiz_lang
    stem = next_item.stem
    options = next_item.options
    if lang != "en":
        t = (next_item.translations or {}).get(lang)  # type: ignore[arg-type]
        if t:
            stem = t.stem
            options = t.options

    st.markdown(f"**Progress:** {len(sess.answers)} / {sess.total_questions}")
    st.progress(len(sess.answers) / sess.total_questions)

    st.markdown("---")
    st.markdown(f"### Q{len(sess.answers) + 1}")
    st.markdown(
        f"**Skill:** {next_item.skill}  \n**Difficulty:** {next_item.difficulty_label}  \n**Bloom:** {next_item.bloom_level}"
    )
    st.write(stem)

    # FIX: no preselected option (all circles same until user clicks)
    radio_key = f"choice_{sess.step_index}"
    choice: Optional[int]
    try:
        choice = st.radio(
            "Choose an option:",
            options=list(range(4)),
            index=None,
            format_func=lambda idx: f"{chr(65 + idx)}. {options[idx]}",
            key=radio_key,
        )
    except TypeError:
        # older streamlit fallback
        choice = st.radio(
            "Choose an option:",
            options=list(range(4)),
            format_func=lambda idx: f"{chr(65 + idx)}. {options[idx]}",
            key=radio_key,
        )

    if st.button("Submit answer", type="primary", use_container_width=True):
        if choice is None:
            st.warning("Please select an option before submitting.")
            st.stop()

        was_correct = int(choice) == int(next_item.correct_index)
        sess.apply_answer(item=next_item, chosen_index=int(choice), was_correct=was_correct)

        ensure_data_dirs()
        out_path = os.path.join("data", "sessions", f"{sess.session_id}.json")
        save_json_atomic(out_path, sess.to_json_dict())
        st.session_state.latest_session_path = out_path

        st.toast("Correct ✅" if was_correct else "Incorrect ❌")
        with st.expander("Explanation (English)", expanded=not was_correct):
            st.write(next_item.explanation_short)

        st.rerun()

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Theta", f"{sess.theta:.2f}")
    with c2:
        st.metric("SEM", f"{sess.theta_sem:.2f}")
    with c3:
        st.metric("Answered", len(sess.answers))
    with c4:
        st.metric("Accuracy", f"{(sess.correct_count / max(1, len(sess.answers)) * 100):.0f}%")


def _render_dashboard_tab() -> None:
    st.subheader("3) Diagnostic Dashboard")

    ensure_data_dirs()
    session_files = list_session_files()
    if not session_files:
        st.info("No saved sessions yet. Complete a test in the Test tab.")
        return

    # Download all sessions ZIP (for Streamlit Cloud demo)
    st.download_button(
        "Download all sessions (ZIP)",
        data=_zip_files(session_files),
        file_name="sessions.zip",
        mime="application/zip",
        use_container_width=True,
    )

    default_path = (
        st.session_state.latest_session_path if st.session_state.latest_session_path in session_files else session_files[0]
    )
    selected = st.selectbox(
        "Choose a session",
        options=session_files,
        index=session_files.index(default_path),
        format_func=lambda p: os.path.basename(p),
    )

    sess_data = load_json(selected)
    sess = AdaptiveSession.from_json_dict(sess_data)

    confidence = compute_confidence_heuristic(sess)
    top_mis = summarize_misconceptions(sess, top_k=8)

    st.markdown("#### Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Theta", f"{sess.theta:.2f}")
    with c2:
        st.metric("SEM", f"{sess.theta_sem:.2f}")
    with c3:
        st.metric("Reliability (heuristic)", f"{sess.reliability_heuristic*100:.0f}%")
    with c4:
        st.metric("Answered", len(sess.answers))
    with c5:
        st.metric("Confidence (heuristic)", f"{confidence*100:.0f}%")

    st.markdown("#### Charts")
    st.pyplot(make_theta_fig(sess), clear_figure=True)
    st.pyplot(make_mastery_heatmap_fig(sess), clear_figure=True)
    st.pyplot(make_difficulty_path_fig(sess), clear_figure=True)

    st.markdown("#### Misconceptions")
    if not top_mis:
        st.info("No misconception signals yet.")
    else:
        st.table([{"misconception": k, "count": v} for k, v in top_mis])

    st.markdown("---")
    st.subheader("AI Narrative & Recommendations (ONE LLM call)")

    if not _get_openai_key_present():
        st.warning("OpenAI API key not found. Set env var OPENAI_API_KEY or Streamlit secret OPENAI_API_KEY.")
        return

    if st.button("Generate AI insights", type="primary", use_container_width=True):
        with st.spinner("Generating AI insights..."):
            try:
                ai = generate_ai_insights_one_call(
                    sess=sess,
                    model=st.session_state.model,
                    top_misconceptions=top_mis,
                    confidence=confidence,
                )
                sess.ai_insights = ai
                save_json_atomic(selected, sess.to_json_dict())
                st.success("AI insights generated.")
            except Exception as e:
                st.error(f"AI insights failed: {e}")

    if sess.ai_insights:
        st.markdown("#### Narrative")
        st.write(sess.ai_insights["narrative"])
        st.markdown("#### Recommendations")
        for b in sess.ai_insights["recommendations"]:
            st.markdown(f"- {b}")
    else:
        st.info("Click **Generate AI insights** to create narrative + recommendations.")

    st.markdown("---")
    st.subheader("Bank-level information curve")
    if st.session_state.item_bank is not None:
        st.pyplot(make_bank_info_curve_fig(st.session_state.item_bank), clear_figure=True)
    else:
        st.info("Load/generate an item bank to view bank-level metrics.")

    st.markdown("---")
    st.subheader("Optional: Empirical calibration (2PL JML) from saved sessions")
    st.caption("No LLM calls. Needs multiple sessions to be meaningful.")
    if st.session_state.item_bank is None:
        st.info("Load/generate an item bank first.")
    else:
        min_sessions = st.slider("Min sessions required", 5, 100, 20, 5)
        if st.button("Run calibration and update bank (in-memory)", use_container_width=True):
            bank: ItemBank = st.session_state.item_bank
            with st.spinner("Calibrating..."):
                updated_bank, calib_report = calibrate_2pl_jml_from_sessions(bank, session_files, min_sessions=min_sessions)
            st.session_state.item_bank = updated_bank
            st.success("Calibration complete (in-memory). Save/download bank from Generate tab to persist.")
            st.json(calib_report)

    st.markdown("---")
    st.subheader("Export PDF Report")
    figs = {
        "theta": make_theta_fig(sess),
        "mastery": make_mastery_heatmap_fig(sess),
        "difficulty": make_difficulty_path_fig(sess),
    }

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
            file_name=f"adaptive_report_{os.path.basename(selected).replace('.json','')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    for f in figs.values():
        plt.close(f)


def main() -> None:
    ensure_data_dirs()
    _init_state()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Item generation + translations + psychometric QC + adaptive IRT test + dashboard + PDF export.")

    with st.expander("System status", expanded=False):
        st.write(
            {
                "openai_key_present": _get_openai_key_present(),
                "model_default": DEFAULT_MODEL,
                "current_model": st.session_state.model,
                "item_bank_path": st.session_state.item_bank_path,
                "latest_session_path": st.session_state.latest_session_path,
                "qc_enabled": st.session_state.qc_enabled,
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
