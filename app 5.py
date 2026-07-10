"""
Mental Health Condition Predictor — Dashboard UI
====================================================================

A clean, minimal, dashboard-style Streamlit app. Inputs live in the
sidebar; the main area is a pure results dashboard: KPI cards, a
confidence chart, and a factors panel. Built to be simple, neat, and
easy to read at a glance.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud (streamlit.io):
    Push this file, requirements.txt, and models/ to GitHub, then
    point a new app at app.py.
"""

import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette — quiet, neutral, one accent color
# ---------------------------------------------------------------------------
INK = "#1F2A24"          # near-black text
MUTED = "#6B7A70"        # secondary text
PRIMARY = "#2C5F2D"      # forest green accent
PRIMARY_SOFT = "#E7F0E8"
LINE = "#E7EBE8"         # hairline borders
CARD_BG = "#FFFFFF"
PAGE_BG = "#F7F9F7"

CONDITION_META = {
    "Normal":     {"color": "#2C5F2D", "soft": "#E7F0E8", "icon": "🌿", "risk": "Low",      "risk_color": "#2C5F2D"},
    "Anxiety":    {"color": "#A9791E", "soft": "#FBF2DD", "icon": "🌊", "risk": "Moderate", "risk_color": "#A9791E"},
    "Burnout":    {"color": "#B25C2E", "soft": "#FBEBE0", "icon": "🔥", "risk": "Moderate", "risk_color": "#B25C2E"},
    "Depression": {"color": "#4652A6", "soft": "#E9EBFB", "icon": "🌧️", "risk": "Elevated", "risk_color": "#4652A6"},
}

CONDITION_MESSAGE = {
    "Normal": "Your responses align with a balanced wellbeing profile. Keep up what's working.",
    "Anxiety": "Elevated tension and restlessness stood out in your responses. Small, steady habits help.",
    "Burnout": "High pressure with little recovery time stood out. Rest and boundaries can help rebuild.",
    "Depression": "Low mood and energy stood out in your responses. Consider talking to someone you trust.",
}

# healthy-direction rules, used only for the on-screen "factors" panel
FEATURE_RULES = {
    "sleep_hours": {"label": "Sleep Hours", "good": "high", "healthy_at": 7.5},
    "sleep_quality": {"label": "Sleep Quality", "good": "high", "healthy_at": 6.5},
    "social_media_hours": {"label": "Social Media", "good": "low", "healthy_at": 3.5},
    "academic_work_pressure": {"label": "Work / Study Pressure", "good": "low", "healthy_at": 5.5},
    "physical_activity_days": {"label": "Physical Activity", "good": "high", "healthy_at": 3.5},
    "stress_level": {"label": "Stress Level", "good": "low", "healthy_at": 5.5},
    "anxiety_score": {"label": "Anxiety Score", "good": "low", "healthy_at": 5.5},
    "depression_score": {"label": "Depression Score", "good": "low", "healthy_at": 5.5},
    "work_life_balance": {"label": "Work-Life Balance", "good": "high", "healthy_at": 5.5},
    "mood_score": {"label": "Mood Score", "good": "high", "healthy_at": 5.5},
    "concentration_level": {"label": "Concentration", "good": "high", "healthy_at": 5.5},
    "social_support": {"label": "Social Support", "good": "high", "healthy_at": 5.5},
}

# wellbeing-score weighting: (feature, invert?, scale_max)
WELLBEING_FEATURES = [
    ("sleep_hours", False, 10),
    ("sleep_quality", False, 10),
    ("physical_activity_days", False, 7),
    ("work_life_balance", False, 10),
    ("mood_score", False, 10),
    ("concentration_level", False, 10),
    ("social_support", False, 10),
    ("academic_work_pressure", True, 10),
    ("social_media_hours", True, 10),
    ("stress_level", True, 10),
    ("anxiety_score", True, 10),
    ("depression_score", True, 10),
]

FEATURE_ORDER = [
    "age", "gender", "occupation", "sleep_hours", "sleep_quality",
    "social_media_hours", "academic_work_pressure", "physical_activity_days",
    "stress_level", "anxiety_score", "depression_score", "work_life_balance",
    "mood_score", "concentration_level", "social_support",
]

MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Styling — flat, quiet, card-based
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
    h1, h2, h3, .dash-title {{ font-family: 'Poppins', sans-serif !important; }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{ background-color: {PAGE_BG}; }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1180px; }}

    /* Top bar */
    .dash-header {{
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 1rem; margin-bottom: 1.4rem;
        border-bottom: 1px solid {LINE};
    }}
    .dash-title {{ font-size: 1.55rem; font-weight: 700; color: {INK}; margin: 0; }}
    .dash-subtitle {{ font-size: 0.9rem; color: {MUTED}; margin-top: 0.15rem; }}
    .dash-badge {{
        background: {PRIMARY_SOFT}; color: {PRIMARY}; font-size: 0.78rem; font-weight: 600;
        padding: 0.35rem 0.8rem; border-radius: 999px;
    }}

    /* Generic card */
    .card {{
        background: {CARD_BG}; border: 1px solid {LINE}; border-radius: 14px;
        padding: 1.1rem 1.3rem; height: 100%;
    }}

    /* KPI card */
    .kpi-label {{ font-size: 0.78rem; color: {MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; margin-top: 0.25rem; color: {INK}; }}
    .kpi-sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.15rem; }}

    /* Section title */
    .section-title {{ font-size: 1.02rem; font-weight: 700; color: {INK}; margin-bottom: 0.7rem; }}

    /* Factor rows */
    .factor-row {{ display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px dashed {LINE}; font-size: 0.88rem; }}
    .factor-row:last-child {{ border-bottom: none; }}
    .factor-name {{ color: {INK}; }}
    .factor-tag {{ font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.55rem; border-radius: 6px; }}
    .tag-watch {{ background: #FBEBE0; color: #B25C2E; }}
    .tag-good {{ background: {PRIMARY_SOFT}; color: {PRIMARY}; }}

    /* Empty state */
    .empty-state {{
        text-align: center; padding: 3.2rem 1rem; color: {MUTED};
        background: {CARD_BG}; border: 1px dashed {LINE}; border-radius: 14px;
    }}
    .empty-state .big-icon {{ font-size: 2.4rem; margin-bottom: 0.6rem; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background-color: {CARD_BG}; border-right: 1px solid {LINE}; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

    /* Buttons */
    div.stButton > button, button[kind="primaryFormSubmit"] {{
        background-color: {PRIMARY} !important; color: white !important;
        border-radius: 9px !important; border: none !important;
        padding: 0.55rem 1.2rem !important; font-weight: 600 !important;
        width: 100%;
    }}
    div.stButton > button:hover, button[kind="primaryFormSubmit"]:hover {{ background-color: #204620 !important; }}

    .disclaimer {{
        font-size: 0.82rem; color: {MUTED}; background: {CARD_BG};
        border: 1px solid {LINE}; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 1rem;
    }}
    .app-footer {{ text-align: center; color: {MUTED}; font-size: 0.78rem; margin-top: 1.6rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    with open(MODEL_DIR / "mental_health_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "le_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open(MODEL_DIR / "le_occupation.pkl", "rb") as f:
        le_occupation = pickle.load(f)
    with open(MODEL_DIR / "le_target.pkl", "rb") as f:
        le_target = pickle.load(f)
    return model, le_gender, le_occupation, le_target


# ---------------------------------------------------------------------------
# Sidebar — all inputs live here
# ---------------------------------------------------------------------------
def render_sidebar_form(le_gender, le_occupation):
    with st.sidebar:
        st.markdown("#### 🧠 Your Details")
        st.caption("Fill this in, then generate your dashboard.")

        with st.form("prediction_form"):
            st.markdown("**About you**")
            age = st.slider("Age", 15, 60, 22)
            gender = st.selectbox("Gender", list(le_gender.classes_))
            occupation = st.selectbox("Occupation", list(le_occupation.classes_))

            st.markdown("**Lifestyle**")
            sleep_hours = st.slider("Sleep hours / night", 3.0, 12.0, 7.0, 0.5)
            sleep_quality = st.slider("Sleep quality", 1.0, 10.0, 6.0, 0.5)
            social_media_hours = st.slider("Social media hours / day", 0.0, 10.0, 3.0, 0.5)
            physical_activity_days = st.slider("Active days / week", 0.0, 7.0, 3.0, 1.0)
            work_life_balance = st.slider("Work-life balance", 1.0, 10.0, 5.0, 0.5)
            academic_work_pressure = st.slider("Work / study pressure", 1.0, 10.0, 5.0, 0.5)

            st.markdown("**Mental wellbeing**")
            stress_level = st.slider("Stress level", 1.0, 10.0, 5.0, 0.5)
            anxiety_score = st.slider("Anxiety score", 1.0, 10.0, 5.0, 0.5)
            depression_score = st.slider("Depression score", 1.0, 10.0, 5.0, 0.5)
            mood_score = st.slider("Mood score", 1.0, 10.0, 5.0, 0.5)
            concentration_level = st.slider("Concentration level", 1.0, 10.0, 5.0, 0.5)
            social_support = st.slider("Social support", 1.0, 10.0, 5.0, 0.5)

            st.write("")
            submitted = st.form_submit_button("Generate Dashboard →")

        st.divider()
        st.caption("Random Forest · 94.9% test accuracy · scikit-learn + Streamlit")

    inputs = {
        "age": age, "gender": gender, "occupation": occupation,
        "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
        "social_media_hours": social_media_hours,
        "academic_work_pressure": academic_work_pressure,
        "physical_activity_days": physical_activity_days,
        "stress_level": stress_level, "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "work_life_balance": work_life_balance, "mood_score": mood_score,
        "concentration_level": concentration_level,
        "social_support": social_support,
    }
    return submitted, inputs


def build_feature_vector(inputs, le_gender, le_occupation):
    gender_encoded = le_gender.transform([inputs["gender"]])[0]
    occupation_encoded = le_occupation.transform([inputs["occupation"]])[0]
    row = {**inputs, "gender": gender_encoded, "occupation": occupation_encoded}
    ordered = [row[f] for f in FEATURE_ORDER]
    return np.array([ordered])


def compute_wellbeing_score(inputs):
    total = 0.0
    for feat, invert, scale_max in WELLBEING_FEATURES:
        val = min(inputs[feat], scale_max)
        pct = (val / scale_max) * 100
        total += (100 - pct) if invert else pct
    return round(total / len(WELLBEING_FEATURES))


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        f"""
        <div class="dash-header">
            <div>
                <div class="dash-title">🧠 Mental Health Dashboard</div>
                <div class="dash-subtitle">A quick, private snapshot based on your lifestyle & mood</div>
            </div>
            <div class="dash-badge">Random Forest · 94.9% accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="big-icon">📋</div>
            <div style="font-weight:600; color:#1F2A24; margin-bottom:0.3rem;">No dashboard yet</div>
            <div>Fill in your details in the sidebar, then click <b>Generate Dashboard</b> to see your results here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(condition, probs, classes, inputs):
    meta = CONDITION_META.get(condition, CONDITION_META["Normal"])
    confidence = float(np.max(probs)) * 100
    wellbeing = compute_wellbeing_score(inputs)

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Predicted Condition", f"{meta['icon']} {condition}")
    with k2:
        kpi_card("Model Confidence", f"{confidence:.0f}%")
    with k3:
        kpi_card("Wellbeing Score", f"{wellbeing}/100", "Higher is better")
    with k4:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value" style="color:{meta['risk_color']};">{meta['risk']}</div>
                <div class="kpi-sub">Based on predicted condition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---- Middle row: confidence chart + factors ----
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)
        df = pd.DataFrame({"Condition": classes, "Confidence": probs * 100})
        df = df.sort_values("Confidence", ascending=True)
        colors = [CONDITION_META.get(c, {}).get("color", PRIMARY) for c in df["Condition"]]
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=22)
            .encode(
                x=alt.X("Confidence:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Condition:N", sort=None, title=None),
                color=alt.Color("Condition:N", scale=alt.Scale(domain=list(df["Condition"]), range=colors), legend=None),
                tooltip=["Condition", alt.Tooltip("Confidence:Q", format=".1f")],
            )
            .properties(height=170)
        )
        labels = chart.mark_text(align="left", dx=5, color=INK, fontWeight="bold", fontSize=11).encode(
            text=alt.Text("Confidence:Q", format=".0f")
        )
        st.altair_chart(chart + labels, use_container_width=True)
        st.markdown(
            f'<div style="font-size:0.85rem; color:{MUTED}; margin-top:0.4rem;">{CONDITION_MESSAGE.get(condition, "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Factors</div>', unsafe_allow_html=True)

        rows_html = ""
        scored = []
        for feat, rule in FEATURE_RULES.items():
            val = inputs[feat]
            healthy = (val >= rule["healthy_at"]) if rule["good"] == "high" else (val <= rule["healthy_at"])
            dist = abs(val - rule["healthy_at"])
            scored.append((dist, rule["label"], val, healthy))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, label, val, healthy in scored[:6]:
            tag_class = "tag-good" if healthy else "tag-watch"
            tag_text = "On track" if healthy else "Watch"
            rows_html += (
                f'<div class="factor-row">'
                f'<span class="factor-name">{label} · {val:g}</span>'
                f'<span class="factor-tag {tag_class}">{tag_text}</span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ This dashboard offers a data-driven estimate based on self-reported lifestyle
            patterns only — it is not a clinical diagnosis. If you're struggling, please reach
            out to a licensed mental health professional or someone you trust.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        model, le_gender, le_occupation, le_target = load_artifacts()
    except FileNotFoundError:
        st.error(
            "⚠️ Model artifacts not found. Make sure `models/mental_health_model.pkl`, "
            "`models/le_gender.pkl`, `models/le_occupation.pkl`, and "
            "`models/le_target.pkl` are present in your repository."
        )
        return

    submitted, inputs = render_sidebar_form(le_gender, le_occupation)

    render_header()

    if submitted:
        features = build_feature_vector(inputs, le_gender, le_occupation)
        prediction = model.predict(features)[0]
        condition = le_target.inverse_transform([prediction])[0]
        probs = model.predict_proba(features)[0]
        classes = le_target.inverse_transform(np.arange(len(probs)))
        render_dashboard(condition, probs, classes, inputs)
    else:
        render_empty_state()

    st.markdown(
        '<div class="app-footer">Mental Health Condition Predictor · Random Forest + Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
