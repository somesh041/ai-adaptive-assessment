from __future__ import annotations

import io
from typing import Dict, List, Tuple

import matplotlib.figure
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.adaptive_engine import AdaptiveSession


def _fig_to_png_bytes(fig: matplotlib.figure.Figure, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def build_pdf_report_bytes(
    sess: AdaptiveSession,
    confidence: float,
    top_misconceptions: List[Tuple[str, int]],
    figures: Dict[str, matplotlib.figure.Figure],
    title: str = "Adaptive Assessment Report",
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    y = height - margin

    def draw_line(text: str, size: int = 11, gap: float = 0.22 * inch):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(margin, y, text)
        y -= gap

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Session: {sess.session_id}" + (f"  |  {sess.session_name}" if sess.session_name else ""))
    y -= 0.25 * inch

    draw_line(f"Theta (IRT): {sess.theta:.2f}    SEM: {sess.theta_sem:.2f}    Reliability (heuristic): {sess.reliability_heuristic*100:.0f}%", 11)
    acc = (sess.correct_count / max(1, len(sess.answers))) * 100
    draw_line(f"Answered: {len(sess.answers)} / {sess.total_questions}    Accuracy: {acc:.0f}%    Confidence (heuristic): {confidence*100:.0f}%", 11)

    y -= 0.10 * inch
    draw_line("Per-skill mastery (lowest skills):", 12)
    weak = sorted(sess.mastery.items(), key=lambda kv: (kv[1], kv[0]))[:6]
    for s, m in weak:
        draw_line(f"- {s}: {m:.2f}", 10, gap=0.18 * inch)

    y -= 0.08 * inch
    draw_line("Top misconceptions:", 12)
    if not top_misconceptions:
        draw_line("- (none recorded)", 10, gap=0.18 * inch)
    else:
        for k, v in top_misconceptions[:8]:
            draw_line(f"- {k} (count: {v})", 10, gap=0.18 * inch)

    y -= 0.12 * inch
    if sess.ai_insights:
        draw_line("AI Narrative:", 12)
        narrative = str(sess.ai_insights.get("narrative", "")).strip()
        c.setFont("Helvetica", 10)
        textobj = c.beginText(margin, y)
        textobj.setLeading(12)
        max_chars = 105
        words = narrative.split()
        line = ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 <= max_chars:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        for ln in lines[:8]:
            textobj.textLine(ln)
        c.drawText(textobj)
        y = textobj.getY() - 0.12 * inch

        draw_line("Recommendations:", 12)
        recs = sess.ai_insights.get("recommendations", [])
        for r in recs[:5]:
            draw_line(f"- {r}", 10, gap=0.18 * inch)

    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Charts")
    y -= 0.35 * inch

    def need_space(px: float) -> bool:
        return y - px < margin

    order = [("theta", "Theta over time"), ("mastery", "Mastery heatmap"), ("difficulty", "Difficulty path")]
    for key, label in order:
        fig = figures.get(key)
        if fig is None:
            continue
        if need_space(3.2 * inch):
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, label)
        y -= 0.2 * inch

        png_bytes = _fig_to_png_bytes(fig)
        img_reader = ImageReader(io.BytesIO(png_bytes))
        img_w = width - 2 * margin
        img_h = 2.6 * inch
        c.drawImage(img_reader, margin, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor="n")
        y -= img_h + 0.35 * inch

    c.save()
    return buf.getvalue()
