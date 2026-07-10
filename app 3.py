"""
Mental Health Condition Predictor — Streamlit App (Enhanced UI/UX)
====================================================================

A polished, deploy-ready Streamlit application for the Mental Health
Condition Predictor project. Loads a pre-trained Random Forest model
and its label encoders, walks the user through a friendly, tabbed
questionnaire, and returns a color-coded prediction with a confidence
breakdown and a few plain-language, data-driven insights.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud (streamlit.io):
    1. Push this file, requirements.txt, and the models/ folder to GitHub.
    2. On share.streamlit.io, point a new app at app.py in that repo.
"""

import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Condition Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette (matches the project report / slide deck)
# ---------------------------------------------------------------------------
PRIMARY = "#2C5F2D"       # deep forest green
PRIMARY_DARK = "#1B3D1C"
SECOND = "#84B59F"        # sage
ACCENT = "#C97B4A"        # terracotta
LIGHT_BG = "#F4F8F5"      # soft sage tint
TEXT_DARK = "#20331F"
TEXT_GREY = "#5B6B60"

CONDITION_META = {
    "Normal": {
        "color": "#2C5F2D",
        "bg": "#EAF3EC",
        "icon": "🌿",
        "headline": "You're showing signs of balanced mental wellbeing.",
        "message": (
            "Your responses align closely with a Normal wellbeing profile. "
            "Keep up the habits that are working — consistent sleep, manageable "
            "stress, and strong social support are the biggest protective factors."
        ),
    },
    "Anxiety": {
        "color": "#B08900",
        "bg": "#FBF4DF",
        "icon": "🌊",
        "headline": "Your responses suggest elevated anxiety indicators.",
        "message": (
            "Racing thoughts, tension, and disrupted sleep often travel together. "
            "Small, consistent steps — grounding techniques, regular sleep, and "
            "talking to someone you trust — can help. Consider speaking with a "
            "counselor if this feeling persists."
        ),
    },
    "Burnout": {
        "color": "#C97B4A",
        "bg": "#FBEBE0",
        "icon": "🔥",
        "headline": "Your responses point toward burnout-related exhaustion.",
        "message": (
            "High pressure combined with little recovery time is a classic burnout "
            "pattern. Prioritizing rest, setting boundaries around work or study, "
            "and reconnecting with activities you enjoy can help rebuild your reserves."
        ),
    },
    "Depression": {
        "color": "#4A5FC9",
        "bg": "#E9EBFB",
        "icon": "🌧️",
        "headline": "Your responses suggest depressive indicators worth attention.",
        "message": (
            "Low mood, low energy, and reduced concentration are signals worth "
            "taking seriously. You don't have to navigate this alone — a mental "
            "health professional can help you find the right support."
        ),
    },
}

# Simple, transparent "healthy direction" rules used only for the
# plain-language insights panel — NOT used by the model itself.
FEATURE_RULES = {
    "sleep_hours": {"label": "Sleep Hours", "good": "high", "healthy_at": 7.5},
    "sleep_quality": {"label": "Sleep Quality", "good": "high", "healthy_at": 6.5},
    "social_media_hours": {"label": "Social Media Hours", "good": "low", "healthy_at": 3.5},
    "academic_work_pressure": {"label": "Academic / Work Pressure", "good": "low", "healthy_at": 5.5},
    "physical_activity_days": {"label": "Physical Activity", "good": "high", "healthy_at": 3.5},
    "stress_level": {"label": "Stress Level", "good": "low", "healthy_at": 5.5},
    "anxiety_score": {"label": "Anxiety Score", "good": "low", "healthy_at": 5.5},
    "depression_score": {"label": "Depression Score", "good": "low", "healthy_at": 5.5},
    "work_life_balance": {"label": "Work-Life Balance", "good": "high", "healthy_at": 5.5},
    "mood_score": {"label": "Mood Score", "good": "high", "healthy_at": 5.5},
    "concentration_level": {"label": "Concentration Level", "good": "high", "healthy_at": 5.5},
    "social_support": {"label": "Social Support", "good": "high", "healthy_at": 5.5},
}

FEATURE_ORDER = [
    "age", "gender", "occupation", "sleep_hours", "sleep_quality",
    "social_media_hours", "academic_work_pressure", "physical_activity_days",
    "stress_level", "anxiety_score", "depression_score", "work_life_balance",
    "mood_score", "concentration_level", "social_support",
]

MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .hero-title {{
        font-family: 'Poppins', sans-serif !important;
    }}

    /* Hide default Streamlit chrome for a cleaner deployed look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }}

    /* Hero banner */
    .hero {{
        background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.6rem;
        color: white;
    }}
    .hero-kicker {{
        color: {SECOND};
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        line-height: 1.25;
    }}
    .hero-subtitle {{
        color: #D7E5DA;
        font-size: 1.02rem;
        max-width: 640px;
        line-height: 1.5;
    }}

    /* Section card */
    .section-card {{
        background: {LIGHT_BG};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }}

    /* Result card */
    .result-card {{
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 0.6rem 0 1.2rem 0;
        border: 1px solid rgba(0,0,0,0.05);
    }}
    .result-icon {{
        font-size: 2.6rem;
        line-height: 1;
    }}
    .result-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.4rem 0 0.2rem 0;
    }}
    .result-headline {{
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .result-message {{
        color: {TEXT_GREY};
        font-size: 0.96rem;
        line-height: 1.55;
    }}

    /* Insight chips */
    .chip {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem 0.35rem 0.2rem 0;
    }}
    .chip-warn {{ background: #FBEBE0; color: {ACCENT}; }}
    .chip-good {{ background: #EAF3EC; color: {PRIMARY}; }}

    /* Buttons */
    div.stButton > button, button[kind="primaryFormSubmit"] {{
        background-color: {PRIMARY} !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out;
    }}
    div.stButton > button:hover, button[kind="primaryFormSubmit"]:hover {{
        background-color: {PRIMARY_DARK} !important;
        transform: translateY(-1px);
    }}

    /* Disclaimer footer */
    .disclaimer {{
        background: #FBEBE0;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: {TEXT_DARK};
        margin-top: 1rem;
    }}
    .app-footer {{
        text-align: center;
        color: {TEXT_GREY};
        font-size: 0.8rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model artifacts (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
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
# UI sections
# ---------------------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-Powered Self-Awareness Tool</div>
            <div class="hero-title">🧠 Mental Health Condition Predictor</div>
            <div class="hero-subtitle">
                Answer a few quick questions about your sleep, habits, and mood.
                A Random Forest model trained on real lifestyle data will estimate
                which wellbeing profile best matches your responses — instantly,
                and completely private to this session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 About this tool")
        st.write(
            "This app uses a **Random Forest Classifier** trained on demographic, "
            "lifestyle, and psychological survey data to estimate a likely mental "
            "health condition: **Normal, Anxiety, Burnout,** or **Depression**."
        )
        st.markdown("### 📊 Model snapshot")
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", "94.9%")
        c2.metric("Classes", "4")
        st.markdown("### ⚙️ How it works")
        st.write(
            "1. Fill in the three question tabs\n"
            "2. Click **Analyze My Responses**\n"
            "3. Get an instant prediction with a confidence breakdown"
        )
        st.divider()
        st.caption(
            "Built with scikit-learn + Streamlit. This tool offers a data-driven "
            "estimate only and is not a clinical diagnostic instrument."
        )


def render_input_form(le_gender, le_occupation):
    with st.form("prediction_form"):
        tab1, tab2, tab3 = st.tabs(["👤 About You", "🌙 Lifestyle", "💭 Mental Wellbeing"])

        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            age = c1.slider("Age", 15, 60, 22)
            gender = c2.selectbox("Gender", list(le_gender.classes_))
            occupation = c3.selectbox("Occupation", list(le_occupation.classes_))
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                sleep_hours = st.slider("😴 Sleep Hours (per night)", 3.0, 12.0, 7.0, 0.5)
                sleep_quality = st.slider("✨ Sleep Quality", 1.0, 10.0, 6.0, 0.5, help="1 = very poor, 10 = excellent")
                physical_activity_days = st.slider("🏃 Physical Activity (days / week)", 0.0, 7.0, 3.0, 1.0)
            with c2:
                social_media_hours = st.slider("📱 Social Media Hours (per day)", 0.0, 10.0, 3.0, 0.5)
                academic_work_pressure = st.slider("📚 Academic / Work Pressure", 1.0, 10.0, 5.0, 0.5, help="1 = minimal, 10 = overwhelming")
                work_life_balance = st.slider("⚖️ Work-Life Balance", 1.0, 10.0, 5.0, 0.5, help="1 = poor, 10 = excellent")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                stress_level = st.slider("😣 Stress Level", 1.0, 10.0, 5.0, 0.5)
                anxiety_score = st.slider("😰 Anxiety Score", 1.0, 10.0, 5.0, 0.5)
                depression_score = st.slider("😔 Depression Score", 1.0, 10.0, 5.0, 0.5)
            with c2:
                mood_score = st.slider("🙂 Mood Score", 1.0, 10.0, 5.0, 0.5, help="1 = very low, 10 = very positive")
                concentration_level = st.slider("🎯 Concentration Level", 1.0, 10.0, 5.0, 0.5)
                social_support = st.slider("🤝 Social Support", 1.0, 10.0, 5.0, 0.5, help="1 = isolated, 10 = strongly supported")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        submitted = st.form_submit_button("🔍 Analyze My Responses", use_container_width=True, type="primary")

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


def render_result(condition, probs, classes, inputs):
    meta = CONDITION_META.get(condition, CONDITION_META["Normal"])

    st.markdown("## Your Result")
    st.markdown(
        f"""
        <div class="result-card" style="background:{meta['bg']}; border-left: 6px solid {meta['color']};">
            <div class="result-icon">{meta['icon']}</div>
            <div class="result-title" style="color:{meta['color']};">Predicted Condition: {condition}</div>
            <div class="result-headline" style="color:{TEXT_DARK};">{meta['headline']}</div>
            <div class="result-message">{meta['message']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Confidence Breakdown")
        df = pd.DataFrame({"Condition": classes, "Confidence": probs * 100})
        df = df.sort_values("Confidence", ascending=True)
        colors = [CONDITION_META.get(c, {}).get("color", PRIMARY) for c in df["Condition"]]
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=26)
            .encode(
                x=alt.X("Confidence:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Condition:N", sort=None, title=None),
                color=alt.Color("Condition:N", scale=alt.Scale(domain=list(df["Condition"]), range=colors), legend=None),
                tooltip=["Condition", alt.Tooltip("Confidence:Q", format=".1f")],
            )
            .properties(height=180)
        )
        text = chart.mark_text(align="left", dx=5, color=TEXT_DARK, fontWeight="bold").encode(
            text=alt.Text("Confidence:Q", format=".1f")
        )
        st.altair_chart(chart + text, use_container_width=True)

    with right:
        st.markdown("#### What Stood Out")
        insights = []
        for feat, rule in FEATURE_RULES.items():
            val = inputs[feat]
            healthy = (val >= rule["healthy_at"]) if rule["good"] == "high" else (val <= rule["healthy_at"])
            insights.append((feat, rule["label"], val, healthy))

        concerning = [i for i in insights if not i[3]]
        positive = [i for i in insights if i[3]]
        concerning.sort(key=lambda x: abs(x[2] - FEATURE_RULES[x[0]]["healthy_at"]), reverse=True)
        positive.sort(key=lambda x: abs(x[2] - FEATURE_RULES[x[0]]["healthy_at"]), reverse=True)

        if concerning:
            st.markdown("**Areas to watch:**")
            chips = "".join(f'<span class="chip chip-warn">⚠️ {label}: {val:g}</span>' for _, label, val, _ in concerning[:4])
            st.markdown(chips, unsafe_allow_html=True)
        if positive:
            st.markdown("**Working in your favor:**")
            chips = "".join(f'<span class="chip chip-good">✓ {label}: {val:g}</span>' for _, label, val, _ in positive[:4])
            st.markdown(chips, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <strong>This is not a clinical diagnosis.</strong> This tool offers a
            data-driven estimate based on self-reported lifestyle patterns only.
            If you're struggling, please reach out to a licensed mental health
            professional or a trusted person in your life.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    render_hero()

    try:
        model, le_gender, le_occupation, le_target = load_artifacts()
    except FileNotFoundError:
        st.error(
            "⚠️ Model artifacts not found. Make sure `models/mental_health_model.pkl`, "
            "`models/le_gender.pkl`, `models/le_occupation.pkl`, and "
            "`models/le_target.pkl` are present in your repository."
        )
        return

    submitted, inputs = render_input_form(le_gender, le_occupation)

    if submitted:
        features = build_feature_vector(inputs, le_gender, le_occupation)
        prediction = model.predict(features)[0]
        condition = le_target.inverse_transform([prediction])[0]
        probs = model.predict_proba(features)[0]
        classes = le_target.inverse_transform(np.arange(len(probs)))
        render_result(condition, probs, classes, inputs)
    else:
        st.info("👆 Fill in the three tabs above and click **Analyze My Responses** to see your result.")

    st.markdown(
        '<div class="app-footer">Mental Health Condition Predictor · Random Forest + Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
