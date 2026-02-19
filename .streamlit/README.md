# AI-Driven Adaptive Assessment & Diagnostic Intelligence System (Streamlit)

A production-ready Streamlit repo that demonstrates:

- **Item Bank Generator** (LLM-assisted, strict JSON, Pydantic validation + auto-repair)
- **Adaptive Test Engine** (**NO LLM calls during quiz**)
- **Diagnostic Dashboard** (theta, confidence heuristic, mastery heatmap, difficulty path, misconceptions)
- **AI narrative + recommendations** (ONE LLM call)
- **PDF export** (ReportLab)

---

## 1) Local setup

### Prereqs
- Python 3.10+ (recommended 3.11)
- An OpenAI API key

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
