# AI-Driven Adaptive Assessment & Diagnostic Intelligence System (Streamlit)

This repo demonstrates a production-ready pipeline for:

- **AI item generation** (OpenAI Responses API, strict JSON, Pydantic validation + auto-repair)
- **Psychometric QC** (construct alignment checks + item-writing flaw detection + IRT priors + auto-regeneration)
- **Optional Indian language translations** (Hindi/Marathi/Odia/Telugu) saved into the **same JSON**
- **Adaptive test engine with NO LLM calls during quiz**
  - IRT-based theta update (2PL/3PL-style)
  - Cognitive mastery update using **BKT** (per-skill P(known))
- **Diagnostic dashboard**
- **PDF export**

## Local run

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
