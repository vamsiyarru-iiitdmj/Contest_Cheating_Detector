"""
app.py -- Contest Cheating Detector (single-process Streamlit app)

Runs entirely inside one Streamlit process -- no FastAPI backend, no
Docker, no separate uvicorn server. Everything api.py used to do over
HTTP now happens as in-process calls into model_utils.py, so this
deploys on the free "Streamlit SDK" Hugging Face Space (packages.txt
still needed for the graphviz system binary dtreeviz shells out to).

All non-UI logic (model loading, prediction, dtreeviz explanation,
gauge chart) lives in model_utils.py, which is independently unit-tested
since it has no Streamlit calls in it. This file is just the UI.
"""

import streamlit as st

from model_utils import (
    FEATURE_COLUMN_ORDER,
    FEATURE_DESCRIPTIONS,
    RISK_COLOR,
    build_viz_model,
    load_models,
    probability_gauge,
    run_prediction,
)
import pandas as pd


@st.cache_resource(show_spinner="Loading models...")
def cached_models():
    return load_models()


@st.cache_resource(show_spinner="Preparing explanation engine...")
def cached_viz_model(_dt):
    # Leading underscore on _dt tells st.cache_resource not to hash the
    # sklearn model object itself (it isn't hashable in a useful way).
    try:
        return build_viz_model(_dt)
    except Exception as exc:
        st.session_state["_viz_build_error"] = str(exc)
        return None


st.set_page_config(page_title="Contest Cheating Detector", layout="wide")

st.markdown(
    """
    <style>
    .cd-bubble {
        background:#2f6fed; color:white; font-weight:700; font-size:1.1rem;
        padding:14px 22px; border-radius:14px; display:inline-block;
        clip-path: polygon(
            2% 10%, 8% 2%, 15% 8%, 22% 1%, 30% 9%, 38% 2%, 46% 9%, 55% 2%,
            63% 9%, 71% 2%, 79% 9%, 88% 2%, 96% 9%, 100% 20%, 97% 35%,
            100% 50%, 96% 65%, 100% 80%, 94% 92%, 86% 98%, 78% 91%,
            70% 98%, 62% 91%, 54% 98%, 46% 91%, 38% 98%, 30% 91%,
            22% 98%, 14% 91%, 6% 98%, 2% 88%, 5% 75%, 1% 60%, 5% 45%,
            1% 30%, 5% 18%
        );
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, about_col, help_col = st.columns([6, 1, 1])
with title_col:
    st.markdown("## 🏆 CONTEST CHEATING DETECTOR")
with about_col:
    with st.popover("About"):
        st.markdown(
            "This app was buit to **detect cheating** and also the **type of person** he is.\n\n"
            "**Academic purpose.** Uses machine learning to analyze "
            "submission patterns for educational insight.\n\n"
            "Confidentially maintained."
        )
with help_col:
    with st.popover("Help"):
        st.markdown(
            "**Usage instructions**\n\n"
            "Upload your `submission_history.csv` export, fill in the "
            "contest details below, then click **Predict!**\n\n"
            "**Data requirements**\n\n"
            "The CSV needs at least a `date` and `publicScore` column, "
            "exported directly from your submission history.\n\n"
            "**Steps to get your submission history**\n\n"
            "1) `pip install kaggle` in cmd"
            "2) Get API from Kaggle (from settings -> Create API -> Copy the API token)\n\n"
            "3) Now, paste the api token in cmd : `set` instead of `export` for  windows.\n\n"
            "4) Finally, paste this command\n\n`kaggle competitions submissions -c summer-of-ml-26-regression-hacakthon -v > file_name.csv`\n\n"
            "5) Your .csv file gets downloaded in your present working directory.\n\nSubmit Here To Predict."
        )

st.divider()

st.markdown('<div class="cd-bubble">Ready to Prove Yourself!! 🔍</div>', unsafe_allow_html=True)

with st.expander("📌 Note"):
    st.markdown(
        "- `claude_min_score` is Claude's own **worst** RMSE submission on this exact contest.\n"
        "- `Rank` and `Quiz Attendance` come from the contest leaderboard and the summer-training "
        "quiz tracker respectively -- they aren't in your submission CSV, so please enter them "
        "accurately; they materially affect the prediction.\n"
        "- Contest timing (start date / duration) can be left at the defaults if you're not sure; "
        "the app will fall back sensibly and tell you what it approximated.\n"
        "- Predictions may not be correct all the times, so we also offer the explanation for the classification."
    )

st.markdown("### Submitter Analysis Input")

uploaded_file = st.file_uploader(
    "Upload your submission_history.csv", type=["csv"], label_visibility="visible"
)
upload_color = "#1e7e34" if uploaded_file is not None else "#c0392b"
st.markdown(
    f"""<style>
    [data-testid="stFileUploaderDropzone"] {{
        border: 2px solid {upload_color} !important;
    }}
    </style>""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    final_rank = st.number_input("Rank", min_value=1, value=1, step=1)
    quiz_participation = st.number_input(
        "Quiz Attendance (0-10)", min_value=0, max_value=10, value=5, step=1
    )
with c2:
    claude_min_score = st.number_input(
        "Claude's reference worst score (claude_min_score)",
        min_value=0.0,
        value=0.60,
        step=0.01,
        format="%.4f",
    )
    total_participants = st.number_input("Total Participants", min_value=1, value=30, step=1)

with st.expander("Contest timing (advanced)"):
    tc1, tc2 = st.columns(2)
    with tc1:
        competition_start = st.text_input(
            "Contest start (YYYY-MM-DD HH:MM:SS, blank = use first submission)",
            value="",
        )
    with tc2:
        competition_duration_days = st.number_input(
            "Contest duration (days)", min_value=1, value=10, step=1
        )

predict_clicked = st.button("Predict!", type="primary")

st.caption("Results and detailed analysis are displayed below")
st.divider()

# --------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------- #
if predict_clicked:
    if uploaded_file is None:
        st.error("Please upload a submission_history.csv file first.")
    elif final_rank > total_participants:
        st.error("Rank can't be greater than Total Participants -- please check these values.")
    else:
        xgb, dt = cached_models()
        viz_model = cached_viz_model(dt)

        preproc_kwargs = dict(
            claude_min_score=claude_min_score,
            quiz_participation=quiz_participation,
            competition_duration_days=competition_duration_days,
            final_rank=final_rank,
            total_participants=total_participants,
        )
        if competition_start.strip():
            preproc_kwargs["competition_start"] = competition_start.strip()

        with st.spinner("Analyzing submission pattern..."):
            try:
                result = run_prediction(
                    uploaded_file.getvalue(), xgb, dt, viz_model, **preproc_kwargs
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Could not process this file: {exc}")

# --------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------- #
result = st.session_state.get("last_result")
if result:
    st.markdown("### Classification Results")

    rc1, rc2 = st.columns(2)

    with rc1:
        xgb_r = result["xgboost"]
        color = RISK_COLOR.get(xgb_r["risk_tier"], "#555")
        st.markdown("**XGBoost Classification**")
        st.markdown(
            f"Character: <span style='background-color:{color};color:white;"
            f"padding:2px 10px;border-radius:12px;font-weight:600;'>"
            f"{xgb_r['archetype']}</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(probability_gauge(xgb_r["top_probability"]), use_container_width=True)
        with st.expander("Full probability breakdown"):
            proba_df = (
                pd.DataFrame(xgb_r["probabilities"].items(), columns=["Archetype", "Probability"])
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(proba_df, hide_index=True, use_container_width=True)

    with rc2:
        dt_r = result["decision_tree"]
        color = RISK_COLOR.get(dt_r["risk_tier"], "#555")
        st.markdown("**Decision Tree Classifier**")
        st.markdown(
            f"Character: <span style='background-color:{color};color:white;"
            f"padding:2px 10px;border-radius:12px;font-weight:600;'>"
            f"{dt_r['archetype']}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Reason for the prediction is given below.")

    st.divider()

    st.markdown("### Explanation for Decision Tree Classification")
    st.caption("A visualisation from dtreeviz -- user's features lie in:")
    explanation = result.get("explanation", {})
    svg = explanation.get("svg")
    if svg:
        st.components.v1.html(f"<div style='overflow-x:auto'>{svg}</div>", height=450, scrolling=True)
    else:
        st.info("Visual explanation unavailable for this input.")
    if explanation.get("text"):
        with st.expander("Explain prediction path (text)"):
            st.text(explanation["text"])

    for w in result.get("warnings", []):
        st.warning(w)

    with st.expander("Require Features Extracted", expanded=False):
        row = result["row"].iloc[0]
        feat_df = pd.DataFrame(
            [
                {
                    "Feature Name": col,
                    "Value": round(float(row[col]), 4) if pd.notna(row[col]) else row[col],
                    "Description": FEATURE_DESCRIPTIONS.get(col, ""),
                }
                for col in FEATURE_COLUMN_ORDER
            ]
        )
        st.dataframe(feat_df, hide_index=True, use_container_width=True)
