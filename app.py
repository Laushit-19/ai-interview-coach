"""
AI Interview Coach — Phase 4+: Streamlit App (upgraded)

Adds on top of the core scoring engine:
    - Filler-word & answer-quality analysis
    - Adaptive follow-up questions when you score poorly on a topic
    - Session history across runs + a progress trend chart
    - Daily practice streak
    - Downloadable PDF report of your results

Run with:  streamlit run app.py
(questions.json must be in the same folder as this file)

Note on persistence: session history and streak are saved to local JSON files
next to this script. This works reliably when run locally. If deployed to a
free-tier host with an ephemeral filesystem, these files may reset when the
app restarts/sleeps — mentioned here so it's a documented limitation rather
than a surprise, and a good "future work" point (swap for a real database).
"""

import json
import os
import random
import re
import string
import textwrap
from collections import defaultdict
from datetime import date, datetime

import nltk
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF, XPos, YPos
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HISTORY_FILE = "session_history.json"
STREAK_FILE = "streak.json"

# ---------------------------------------------------------------------------
# One-time NLTK setup
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        nltk.download(pkg, quiet=True)
    return True


setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# Scoring engine (unchanged logic from Phase 1/2/3)
# ---------------------------------------------------------------------------
def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]


def stem_tokens(text: str) -> set[str]:
    return {stemmer.stem(tok) for tok in preprocess(text)}


def stems_match(stem_a: str, stem_b: str, min_prefix: int = 5) -> bool:
    if stem_a == stem_b:
        return True
    shortest = min(len(stem_a), len(stem_b))
    prefix_len = min(min_prefix, shortest)
    return shortest >= min_prefix and stem_a[:prefix_len] == stem_b[:prefix_len]


def compute_similarity_score(user_answer: str, model_answer: str) -> float:
    user_clean = " ".join(preprocess(user_answer))
    model_clean = " ".join(preprocess(model_answer))
    if not user_clean.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([model_clean, user_clean])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def keyword_coverage(user_answer: str, keywords: list[str]) -> float:
    user_stems = stem_tokens(user_answer)
    if not keywords:
        return 1.0
    present = sum(
        1
        for kw in keywords
        if any(stems_match(stemmer.stem(kw.lower()), us) for us in user_stems)
    )
    return present / len(keywords)


def find_missing_keywords(user_answer: str, keywords: list[str]) -> list[str]:
    user_stems = stem_tokens(user_answer)
    return [
        kw
        for kw in keywords
        if not any(stems_match(stemmer.stem(kw.lower()), us) for us in user_stems)
    ]


def compute_final_score(user_answer: str, model_answer: str, keywords: list[str]) -> float:
    tfidf_score = compute_similarity_score(user_answer, model_answer)
    coverage_score = keyword_coverage(user_answer, keywords) * 100
    return round(0.5 * tfidf_score + 0.5 * coverage_score, 2)


def get_score_band(score: float) -> str:
    if score >= 75:
        return "Excellent"
    elif score >= 55:
        return "Good"
    elif score >= 35:
        return "Average"
    else:
        return "Needs work"


BAND_STYLES = {
    "Excellent": {"ink": "#1F4D36", "bg": "#E4EFE8", "border": "#2F6F4E"},
    "Good": {"ink": "#1F4D36", "bg": "#E9F1EA", "border": "#4C8F6A"},
    "Average": {"ink": "#7A4E10", "bg": "#F7EAD3", "border": "#C98A2B"},
    "Needs work": {"ink": "#7A2E1B", "bg": "#F4E0D8", "border": "#B4472F"},
}


def generate_feedback(score: float, missing: list[str]) -> str:
    if score >= 75:
        msg = "Strong answer — you covered the key concepts clearly."
    elif score >= 55:
        msg = "Solid answer, with room to add more depth."
    elif score >= 35:
        msg = "You're on the right track, but some important ideas are missing."
    else:
        msg = "This needs significant work — try to address the core concept directly."

    if missing:
        msg += f" Consider mentioning: {', '.join(missing)}."
    else:
        msg += " You covered every key term we were looking for."
    return msg


# ---------------------------------------------------------------------------
# NEW: Filler-word & answer-quality analysis
# ---------------------------------------------------------------------------
FILLER_WORDS = [
    "um", "uh", "like", "actually", "basically", "literally",
    "you know", "sort of", "kind of", "stuff", "things", "i mean",
]


def analyze_answer_quality(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)

    padded = f" {text.lower()} "
    filler_counts = {}
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, padded))
        if count > 0:
            filler_counts[filler] = count

    total_fillers = sum(filler_counts.values())

    if word_count < 12:
        length_note = "Your answer is quite brief — consider adding a specific example or more detail."
    elif word_count > 180:
        length_note = "Your answer is fairly long — in a real interview, aim to be more concise."
    else:
        length_note = None

    return {
        "word_count": word_count,
        "filler_counts": filler_counts,
        "total_fillers": total_fillers,
        "length_note": length_note,
    }


# ---------------------------------------------------------------------------
# NEW: local persistence helpers (session history + streak)
# ---------------------------------------------------------------------------
def load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json_file(path: str, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def record_session_history(overall_avg: float, category_avgs: dict, num_questions: int) -> None:
    history = load_json_file(HISTORY_FILE, [])
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "overall_avg": overall_avg,
            "category_avgs": category_avgs,
            "num_questions": num_questions,
        }
    )
    save_json_file(HISTORY_FILE, history)


def update_streak() -> dict:
    today = date.today().isoformat()
    data = load_json_file(STREAK_FILE, {"last_date": None, "current_streak": 0, "longest_streak": 0})

    if data["last_date"] == today:
        return data  # already practiced today, no change

    if data["last_date"] is not None:
        last_date_obj = date.fromisoformat(data["last_date"])
        gap_days = (date.today() - last_date_obj).days
        data["current_streak"] = data["current_streak"] + 1 if gap_days == 1 else 1
    else:
        data["current_streak"] = 1

    data["last_date"] = today
    data["longest_streak"] = max(data.get("longest_streak", 0), data["current_streak"])
    save_json_file(STREAK_FILE, data)
    return data


def get_streak_display() -> dict:
    return load_json_file(STREAK_FILE, {"last_date": None, "current_streak": 0, "longest_streak": 0})


# ---------------------------------------------------------------------------
# NEW: PDF report generation
# ---------------------------------------------------------------------------
def _safe(text: str) -> str:
    """fpdf's core fonts only support latin-1; strip anything outside it."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf_paragraph(pdf: FPDF, text: str, width: int = 95, bold: bool = False, size: int = 10) -> None:
    """Write text to the PDF pre-wrapped ourselves (via textwrap), so fpdf's own
    line-breaking never has to handle long/unusual unbroken tokens — this is what
    was triggering 'Not enough horizontal space to render a single character' on
    some fpdf2 versions."""
    pdf.set_font("Helvetica", "B" if bold else "", size)
    for line in (textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]):
        pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_pdf_report(results: list[dict], overall_avg: float, category_avgs: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Interview Coach - Report Card", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe(datetime.now().strftime("Generated on %d %b %Y, %H:%M")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe(f"Overall average: {overall_avg}/100"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for cat, avg in category_avgs.items():
        pdf.cell(0, 7, _safe(f"  - {cat}: {avg}/100"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Question-by-question breakdown", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    for r in results:
        _pdf_paragraph(pdf, _safe(f"[{r['category']}] {r['question']}"), bold=True, size=11)
        _pdf_paragraph(pdf, _safe(f"Your answer: {r['user_answer']}"), size=10)
        _pdf_paragraph(pdf, _safe(f"Score: {r['score']}/100 ({r['band']})"), size=10)
        if r["missing_keywords"]:
            _pdf_paragraph(pdf, _safe(f"Missing keywords: {', '.join(r['missing_keywords'])}"), size=10)
        pdf.ln(3)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------
@st.cache_data
def load_question_bank():
    with open("questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


question_bank = load_question_bank()
categories = sorted(set(q["category"] for q in question_bank))
questions_by_id = {q["id"]: q for q in question_bank}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "session_started": False,
        "session_questions": [],
        "current_index": 0,
        "results": [],
        "answered": False,
        "last_score": None,
        "last_band": None,
        "last_missing": None,
        "last_quality": None,
        "followup_note": None,
        "remediated_categories": set(),
        "used_question_ids": set(),
        "history_recorded": False,
        "mock_mode": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def start_session(n_questions: int, category_filter: str, mock_mode: bool = False):
    pool = [
        q
        for q in question_bank
        if category_filter == "All categories" or q["category"] == category_filter
    ]
    n = min(n_questions, len(pool))
    chosen = random.sample(pool, n)
    st.session_state.session_questions = chosen
    st.session_state.current_index = 0
    st.session_state.results = []
    st.session_state.session_started = True
    st.session_state.answered = False
    st.session_state.remediated_categories = set()
    st.session_state.used_question_ids = {q["id"] for q in chosen}
    st.session_state.history_recorded = False
    st.session_state.mock_mode = mock_mode


def restart():
    st.session_state.session_started = False
    st.session_state.session_questions = []
    st.session_state.current_index = 0
    st.session_state.results = []
    st.session_state.answered = False
    st.session_state.remediated_categories = set()
    st.session_state.used_question_ids = set()
    st.session_state.history_recorded = False


def maybe_insert_followup(question: dict, band: str) -> str | None:
    """If the answer was weak, insert an easier same-category question right after this one."""
    category = question["category"]
    if band != "Needs work" or category in st.session_state.remediated_categories:
        return None

    candidates = [
        q
        for q in question_bank
        if q["category"] == category
        and q.get("difficulty") == "basic"
        and q["id"] not in st.session_state.used_question_ids
    ]
    if not candidates:
        return None

    followup = random.choice(candidates)
    insert_at = st.session_state.current_index + 1
    st.session_state.session_questions.insert(insert_at, followup)
    st.session_state.used_question_ids.add(followup["id"])
    st.session_state.remediated_categories.add(category)
    return followup["question"]


def submit_answer(user_answer: str, question: dict):
    score = compute_final_score(user_answer, question["model_answer"], question["keywords"])
    missing = find_missing_keywords(user_answer, question["keywords"])
    band = get_score_band(score)
    quality = analyze_answer_quality(user_answer)

    st.session_state.results.append(
        {
            "id": question["id"],
            "category": question["category"],
            "question": question["question"],
            "user_answer": user_answer,
            "score": score,
            "band": band,
            "missing_keywords": missing,
        }
    )
    st.session_state.answered = True
    st.session_state.last_score = score
    st.session_state.last_band = band
    st.session_state.last_missing = missing
    st.session_state.last_quality = quality
    if st.session_state.mock_mode:
        st.session_state.followup_note = None
    else:
        st.session_state.followup_note = maybe_insert_followup(question, band)


def next_question():
    st.session_state.current_index += 1
    st.session_state.answered = False
    st.session_state.followup_note = None


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--bg:#090D1A;--panel:#11182A;--panel2:#18223A;--border:rgba(255,255,255,.09);--text:#F8FAFC;--muted:#94A3B8;--p:#8B5CF6;--b:#38BDF8;--g:#22C55E;--o:#F59E0B;--r:#F87171}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(139,92,246,.16),transparent 27%),radial-gradient(circle at 90% 5%,rgba(56,189,248,.10),transparent 25%),var(--bg);color:var(--text)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text)}
h1,h2,h3{font-family:'Manrope',sans-serif!important;color:var(--text)!important;font-weight:800!important;letter-spacing:-.03em}
.block-container{max-width:1380px;padding-top:2rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#10172A,#090D1A);border-right:1px solid var(--border)}
section[data-testid="stSidebar"]>div{padding-top:1.3rem}
section[data-testid="stSidebar"]:before{content:"🎯  AI INTERVIEW COACH";display:block;font-family:'Manrope';font-size:20px;font-weight:800;letter-spacing:-.03em;padding:0 1.2rem 1.4rem;color:#fff}
section[data-testid="stSidebar"]:after{content:"AI-powered interview practice";display:block;color:#7F8AA0;font-size:12px;padding:0 1.2rem;position:absolute;top:58px}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(17,24,42,.88)!important;border:1px solid var(--border)!important;border-radius:22px!important;box-shadow:0 18px 50px rgba(0,0,0,.15)}
div[data-testid="stVerticalBlockBorderWrapper"]>div{padding:18px!important}
.coach-tagline{color:var(--muted)!important;font-size:16px!important;margin-top:-4px!important;margin-bottom:20px!important}
.streak-badge{background:linear-gradient(90deg,rgba(245,158,11,.13),rgba(139,92,246,.12))!important;border:1px solid rgba(245,158,11,.25)!important;color:#FCD34D!important;border-radius:999px!important;padding:8px 15px!important;font-weight:700!important}
.coach-tag{background:rgba(139,92,246,.12)!important;border:1px solid rgba(139,92,246,.3)!important;color:#BCA7FF!important;border-radius:999px!important;padding:6px 12px!important;font-weight:700!important}
.coach-tag-followup{background:rgba(245,158,11,.12)!important;border-color:rgba(245,158,11,.3)!important;color:#FCD34D!important}
.coach-question{font-family:'Manrope',sans-serif!important;font-size:28px!important;line-height:1.4!important;color:#fff!important;letter-spacing:-.025em!important}
.dot-row{gap:8px!important}.dot{width:9px!important;height:9px!important;background:#27324A!important;border:0!important}.dot-done{background:#22C55E!important}.dot-current{background:#8B5CF6!important;box-shadow:0 0 0 5px rgba(139,92,246,.14)!important}
.stamp{font-family:'Manrope',sans-serif!important;border-radius:16px!important;padding:14px 22px!important;transform:none!important;font-size:17px!important;box-shadow:0 10px 30px rgba(0,0,0,.12)}
.feedback-text{background:rgba(255,255,255,.035);border:1px solid var(--border);border-radius:14px;padding:16px;color:#CBD5E1!important;font-size:15px!important;line-height:1.65!important}
.quality-line{border-top:0!important;background:rgba(56,189,248,.07);border-left:3px solid #38BDF8;border-radius:8px;padding:10px 12px!important;color:#CBD5E1!important}
.followup-note{background:rgba(245,158,11,.10)!important;border:1px solid rgba(245,158,11,.25)!important;color:#FCD34D!important;border-radius:12px!important;padding:12px 14px!important}
.report-title{font-family:'Manrope',sans-serif!important;font-size:46px!important;background:linear-gradient(90deg,#A78BFA,#38BDF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.report-sub{color:var(--muted)!important}
.bar-label{color:#D8E0EB!important;font-weight:600!important}.bar-track{background:#222D45!important;height:11px!important;border-radius:99px!important}.bar-fill{border-radius:99px!important}.bar-score{color:#CBD5E1!important;font-weight:700!important}
div.stButton>button,div[data-testid="stDownloadButton"]>button{border-radius:12px!important;font-weight:700!important;min-height:44px!important;transition:.2s ease!important}div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#7C3AED,#2563EB)!important;border:0!important;box-shadow:0 8px 24px rgba(124,58,237,.25)!important}div.stButton>button[kind="secondary"]{background:rgba(255,255,255,.035)!important;border:1px solid var(--border)!important;color:#D8E0EB!important}
div.stButton>button:hover{transform:translateY(-1px)}
div[data-testid="stTextArea"] textarea{background:#0D1425!important;color:#F8FAFC!important;border:1px solid rgba(255,255,255,.12)!important;border-radius:14px!important;padding:14px!important}div[data-testid="stTextArea"] textarea:focus{border-color:#8B5CF6!important;box-shadow:0 0 0 3px rgba(139,92,246,.15)!important}
[data-testid="stMetricValue"]{color:#fff!important}.stCaption,small{color:#8F9AAC!important}

/* =========================================
   SIDEBAR NAVIGATION — HIGH CONTRAST + ACTIVE PILL
   ========================================= */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Navigation heading */
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* Each navigation row */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100% !important;
    padding: 10px 12px !important;
    margin: 5px 0 !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all .2s ease !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.06) !important;
    border-color: rgba(255,255,255,.08) !important;
}

/* Force every nav label and nested text to bright white */
section[data-testid="stSidebar"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] div[role="radiogroup"] label *,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #F8FAFC !important;
    fill: #F8FAFC !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Selected navigation item: purple-blue premium pill */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(124,58,237,.92), rgba(37,99,235,.92)) !important;
    border-color: rgba(167,139,250,.55) !important;
    box-shadow: 0 8px 22px rgba(124,58,237,.24) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) *,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Keep secondary sidebar captions readable but visually secondary */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #B7C2D4 !important;
    opacity: 1 !important;
}

/* Main app captions: never allow low-contrast dark text */
.stCaption, small, [data-testid="stCaptionContainer"] p {
    color: #AEB8C8 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


def render_dots(total: int, current_index: int, answered_count: int):
    dots_html = '<div class="dot-row">'
    for i in range(total):
        if i < answered_count:
            cls = "dot dot-done"
        elif i == current_index:
            cls = "dot dot-current"
        else:
            cls = "dot"
        dots_html += f'<div class="{cls}"></div>'
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)


def render_stamp(score: float, band: str):
    style = BAND_STYLES[band]
    st.markdown(
        f'<div class="stamp-wrap"><span class="stamp" '
        f'style="color:{style["ink"]}; background:{style["bg"]}; border-color:{style["border"]};">'
        f"{score}/100 &nbsp;·&nbsp; {band}</span></div>",
        unsafe_allow_html=True,
    )


def render_quality(quality: dict):
    parts = [f"{quality['word_count']} words"]
    if quality["total_fillers"] > 0:
        detail = ", ".join(f"{w} ×{c}" for w, c in quality["filler_counts"].items())
        parts.append(f"{quality['total_fillers']} filler word(s) — {detail}")
    else:
        parts.append("no filler words detected")
    line = " &nbsp;·&nbsp; ".join(parts)
    st.markdown(f'<div class="quality-line">{line}</div>', unsafe_allow_html=True)
    if quality["length_note"]:
        st.caption(quality["length_note"])


def render_countdown_timer(seconds: int = 90, timer_id: str = "t0"):
    """A self-contained, purely visual countdown (doesn't auto-submit — that's
    intentionally kept out to avoid fragile cross-iframe state syncing with
    Streamlit's rerun model). Creates real time pressure without risking a
    forced submission bug during a live demo."""
    html = f"""
    <div id="wrap-{timer_id}" style="
        font-family:'DM Sans',sans-serif; display:flex; align-items:center; gap:12px;
        background:rgba(139,92,246,.08); border:1px solid rgba(139,92,246,.3);
        border-radius:14px; padding:10px 16px; width:fit-content; transition:.3s ease;">
      <span style="font-size:13px; color:#B7C2D4; font-weight:600;">TIME REMAINING</span>
      <span id="txt-{timer_id}" style="font-family:'Manrope',sans-serif; font-size:20px;
        font-weight:800; color:#A78BFA; letter-spacing:.02em;">--:--</span>
    </div>
    <script>
      (function() {{
        let seconds = {seconds};
        const txt = document.getElementById("txt-{timer_id}");
        const wrap = document.getElementById("wrap-{timer_id}");
        function render() {{
          const m = Math.floor(seconds / 60).toString().padStart(2, "0");
          const s = (seconds % 60).toString().padStart(2, "0");
          txt.textContent = m + ":" + s;
          if (seconds <= 10) {{
            txt.style.color = "#F87171"; wrap.style.borderColor = "rgba(248,113,113,.5)";
            wrap.style.background = "rgba(248,113,113,.1)";
          }} else if (seconds <= 30) {{
            txt.style.color = "#F59E0B"; wrap.style.borderColor = "rgba(245,158,11,.5)";
            wrap.style.background = "rgba(245,158,11,.1)";
          }}
        }}
        render();
        const interval = setInterval(function() {{
          seconds -= 1;
          if (seconds <= 0) {{
            txt.textContent = "Time's up";
            clearInterval(interval);
            return;
          }}
          render();
        }}, 1000);
      }})();
    </script>
    """
    components.html(html, height=54)


# ---------------------------------------------------------------------------
# NEW: category icons + tips content
# ---------------------------------------------------------------------------
CATEGORY_ICONS = {
    "Python Basics": "🐍",
    "OOP": "🧩",
    "DSA": "🧮",
    "HR Behavioral": "🤝",
    "Communication": "💬",
}


def get_category_stats() -> dict:
    """Average score per category across all past sessions, from session_history.json."""
    history = load_json_file(HISTORY_FILE, [])
    totals = defaultdict(list)
    for h in history:
        for cat, avg in h.get("category_avgs", {}).items():
            totals[cat].append(avg)
    return {cat: round(sum(vals) / len(vals), 1) for cat, vals in totals.items()}


TIPS_CONTENT = {
    "General": [
        "Practice answering out loud, not just typing — the words that come easily on a page often don't come easily from your mouth.",
        "Aim for 60–90 seconds per spoken answer. Longer than that and interviewers start to lose the thread.",
        "After you get feedback, revisit the missing keywords — that's the fastest way to close real gaps.",
        "Say your main point first, then explain. Don't make the interviewer wait for your conclusion.",
    ],
    "Python Basics": [
        "When asked 'what's the difference between X and Y', explicitly compare both — don't just define one and imply the other.",
        "Back up claims with a tiny example. 'Lists are mutable' is weaker than 'lists are mutable, so `my_list.append()` changes it in place'.",
        "If relevant, mention time/space complexity — it signals you think beyond just 'does it work'.",
    ],
    "OOP": [
        "Anchor answers to the four pillars (encapsulation, inheritance, polymorphism, abstraction) even when not asked directly — it shows structured thinking.",
        "Give a one-line definition first, then a concrete example. Definitions alone sound memorized.",
    ],
    "DSA": [
        "Always state time complexity using Big-O notation, even if not asked — interviewers expect it by default.",
        "Trace through a tiny example out loud. It catches your own mistakes before the interviewer does.",
        "If you don't know the optimal solution, say the brute-force one first. A working answer beats silence.",
    ],
    "HR Behavioral": [
        "Use the STAR method: Situation, Task, Action, Result. It keeps answers concrete instead of vague.",
        "Specific beats general. 'I improved team communication' is weaker than 'I set up a weekly 15-minute sync that cut back-and-forth emails by half'.",
        "It's fine to admit a real weakness — just always pair it with what you're doing about it.",
    ],
    "Communication": [
        "Lead with your main point (BLUF: Bottom Line Up Front), then explain — don't build up to it.",
        "Replace filler words with a short pause. Silence reads as confident; 'um' reads as unsure.",
        "When explaining something technical, check in — ask if that made sense, don't just monologue.",
    ],
}


# ---------------------------------------------------------------------------
# Sidebar navigation + streak badge
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Navigate", ["🎯 Practice", "📈 Progress", "💡 Tips"], label_visibility="collapsed")

streak_data = get_streak_display()
st.markdown("# AI Interview Coach")
st.markdown(
    '<div class="coach-tagline">Practice out loud. Get graded instantly. No manual review.</div>',
    unsafe_allow_html=True,
)
if streak_data["current_streak"] > 0:
    st.markdown(
        f'<div class="streak-badge">🔥 {streak_data["current_streak"]}-day streak '
        f'&nbsp;·&nbsp; best: {streak_data["longest_streak"]}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="streak-badge">🔥 Complete a session today to start your streak</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: History
# ---------------------------------------------------------------------------
if page == "📈 Progress":
    with st.container(border=True):
        st.markdown("### Your progress over time")

        history = load_json_file(HISTORY_FILE, [])
        if not history:
            st.write("No completed sessions yet — finish a practice session to see your trend here.")
        else:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "Session": [datetime.fromisoformat(h["timestamp"]).strftime("%d %b, %H:%M") for h in history],
                    "Overall average": [h["overall_avg"] for h in history],
                }
            ).set_index("Session")
            st.line_chart(df)

            st.write("")
            st.write("**Past sessions:**")
            for h in reversed(history[-10:]):
                ts = datetime.fromisoformat(h["timestamp"]).strftime("%d %b %Y, %H:%M")
                st.write(f"- {ts} — {h['overall_avg']}/100 across {h['num_questions']} questions")

        st.caption(
            "Note: history is saved to a local file next to the app. If deployed to a "
            "free-tier host, this may reset when the app restarts."
        )

# ---------------------------------------------------------------------------
# PAGE: Tips & Tricks
# ---------------------------------------------------------------------------
elif page == "💡 Tips":
    with st.container(border=True):
        st.markdown("### General tips")
        for tip in TIPS_CONTENT["General"]:
            st.markdown(f"- {tip}")

    for cat in categories:
        icon = CATEGORY_ICONS.get(cat, "📌")
        with st.container(border=True):
            st.markdown(f"### {icon} {cat}")
            for tip in TIPS_CONTENT.get(cat, []):
                st.markdown(f"- {tip}")

# ---------------------------------------------------------------------------
# PAGE: Practice — Screen 1: Start
# ---------------------------------------------------------------------------
elif not st.session_state.session_started:
    if "category_filter" not in st.session_state:
        st.session_state.category_filter = "All categories"

    cat_stats = get_category_stats()

    with st.container(border=True):
        st.markdown("### Choose a focus")
        all_options = ["All categories"] + categories
        cols = st.columns(3)
        for i, cat in enumerate(all_options):
            with cols[i % 3]:
                icon = "🎯" if cat == "All categories" else CATEGORY_ICONS.get(cat, "📌")
                avg = cat_stats.get(cat)
                score_line = f"{avg}/100 avg" if avg is not None else "Not attempted yet"
                selected = st.session_state.category_filter == cat
                check = "✓ " if selected else ""
                if st.button(
                    f"{icon} {check}{cat}",
                    key=f"cat_btn_{cat}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.category_filter = cat
                    st.rerun()
                if cat != "All categories":
                    st.caption(score_line)

    with st.container(border=True):
        st.markdown("### Set up your session")
        n_questions = st.slider("Number of questions", min_value=1, max_value=10, value=5)
        mock_mode = st.checkbox(
            "🎬  Mock Interview Mode — timed, feedback revealed only at the end",
            value=False,
            help="Simulates real interview pressure: a visible countdown per question, "
                 "no instant scoring or hints, and no easier follow-ups. You'll see your "
                 "full results only after the last question.",
        )
        st.caption(f"Focus: {st.session_state.category_filter}")
        st.caption(f"Question bank: {len(question_bank)} questions across {len(categories)} categories.")
        if mock_mode:
            st.caption("Mock mode: timer visible, feedback hidden until your report card.")
        else:
            st.caption("Score poorly on a question and the system will slot in an easier follow-up on that topic.")

        if st.button("Begin session", type="primary"):
            start_session(n_questions, st.session_state.category_filter, mock_mode)
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Practice — Screen 2: Active session
# ---------------------------------------------------------------------------
elif st.session_state.current_index < len(st.session_state.session_questions):
    q = st.session_state.session_questions[st.session_state.current_index]
    total = len(st.session_state.session_questions)
    idx = st.session_state.current_index
    is_followup = q.get("difficulty") == "basic" and idx > 0 and st.session_state.results and \
        st.session_state.results[-1]["category"] == q["category"] and \
        st.session_state.results[-1]["band"] == "Needs work"

    with st.container(border=True):
        render_dots(total, idx, len(st.session_state.results))

        tag_class = "coach-tag coach-tag-followup" if is_followup else "coach-tag"
        tag_text = f"{q['category']} · follow-up" if is_followup else q["category"]
        st.markdown(f'<span class="{tag_class}">{tag_text}</span>', unsafe_allow_html=True)
        if st.session_state.mock_mode:
            st.markdown('<span class="coach-tag coach-tag-followup" style="margin-left:8px;">🎬 Mock interview</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="coach-question">{q["question"]}</div>', unsafe_allow_html=True)
        st.write("")

        if st.session_state.mock_mode and not st.session_state.answered:
            render_countdown_timer(seconds=90, timer_id=f"q{idx}")
            st.write("")

        user_answer = st.text_area(
            "Your answer",
            key=f"answer_input_{idx}",
            height=130,
            disabled=st.session_state.answered,
            placeholder="Type your answer here...",
            label_visibility="collapsed",
        )

        if not st.session_state.answered:
            if st.button("Submit answer", type="primary"):
                if not user_answer.strip():
                    st.error("Please type an answer before submitting.")
                else:
                    submit_answer(user_answer, q)
                    st.rerun()
        elif st.session_state.mock_mode:
            st.markdown(
                '<div class="feedback-text">✓ Answer recorded. Scoring and feedback will be '
                'revealed together at the end of the interview.</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            total_now = len(st.session_state.session_questions)
            button_label = "Next question" if idx + 1 < total_now else "Finish interview & see results"
            if st.button(button_label, type="primary"):
                next_question()
                st.rerun()
        else:
            score = st.session_state.last_score
            band = st.session_state.last_band
            missing = st.session_state.last_missing
            quality = st.session_state.last_quality

            render_stamp(score, band)
            st.markdown(f'<div class="feedback-text">{generate_feedback(score, missing)}</div>', unsafe_allow_html=True)
            render_quality(quality)

            if st.session_state.followup_note:
                st.markdown(
                    f'<div class="followup-note">📌 Added a quick follow-up question on this topic: '
                    f'"{st.session_state.followup_note}"</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            total_now = len(st.session_state.session_questions)
            button_label = "Next question" if idx + 1 < total_now else "See report card"
            if st.button(button_label, type="primary"):
                next_question()
                st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Practice — Screen 3: Report card summary
# ---------------------------------------------------------------------------
else:
    results = st.session_state.results
    overall_avg = round(sum(r["score"] for r in results) / len(results), 2)

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["score"])
    category_avgs = {cat: round(sum(s) / len(s), 2) for cat, s in by_category.items()}

    if not st.session_state.history_recorded:
        record_session_history(overall_avg, category_avgs, len(results))
        update_streak()
        st.session_state.history_recorded = True

    with st.container(border=True):
        title_text = "Mock interview results" if st.session_state.mock_mode else "Report card"
        st.markdown(f'<div class="report-title">{title_text}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-sub">Overall average — {overall_avg}/100 across {len(results)} questions</div>',
            unsafe_allow_html=True,
        )

        for cat, avg in sorted(category_avgs.items(), key=lambda kv: -kv[1]):
            if avg >= 75:
                bar_color = "#2F6F4E"
            elif avg >= 55:
                bar_color = "#4C8F6A"
            elif avg >= 35:
                bar_color = "#C98A2B"
            else:
                bar_color = "#B4472F"

            st.markdown(
                f'<div class="bar-row">'
                f'<div class="bar-label">{cat}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{avg}%; background:{bar_color};"></div></div>'
                f'<div class="bar-score">{avg}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        if len(category_avgs) > 1:
            strongest = max(category_avgs, key=category_avgs.get)
            weakest = min(category_avgs, key=category_avgs.get)
            st.write("")
            col1, col2 = st.columns(2)
            col1.success(f"Strongest: {strongest} ({category_avgs[strongest]}/100)")
            col2.warning(f"Focus next: {weakest} ({category_avgs[weakest]}/100)")

        st.write("")
        pdf_bytes = generate_pdf_report(results, overall_avg, category_avgs)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"interview_report_{date.today().isoformat()}.pdf",
            mime="application/pdf",
        )

    with st.container(border=True):
        st.markdown("### Question by question")
        for r in results:
            with st.expander(f"{r['question']}  —  {r['score']}/100 · {r['band']}"):
                st.write(f"**Your answer:** {r['user_answer']}")
                if r["missing_keywords"]:
                    st.write(f"**Missing keywords:** {', '.join(r['missing_keywords'])}")
                else:
                    st.write("All key terms covered.")

    if st.button("Start new session", type="primary"):
        restart()
        st.rerun()
