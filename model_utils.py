"""
model_utils.py -- non-UI logic for the Contest Cheating Detector.

Kept separate from app.py (which has top-level Streamlit calls that only
run inside a live Streamlit server) specifically so this half can be
imported and unit-tested with plain Python, no `streamlit run` required.
"""

import io
import warnings

import joblib
import pandas as pd
import plotly.graph_objects as go

from preprocessor import (
    FEATURE_COLUMN_ORDER,
    extract_features_from_log,
    load_submission_log,
)

# --------------------------------------------------------------------- #
# Label mapping -- must match the {archetype: int} mapping baked into
# start_weights_dt_2.ipynb (cell 3) exactly, since that is the integer
# order both models were fit on. Verified directly against the notebook
# source (not assumed).
# --------------------------------------------------------------------- #
CLASS_NAMES = [
    "hardworker",                    # 0
    "benchmark_tweaker",             # 1
    "daily_sprinter",                # 2
    "consistent_late_breakthrough",  # 3
    "low_interest_quitter",          # 4
    "consistent_ai_leap",            # 5
    "ai_dump_sincere_start",         # 6
    "late_joiner_dangerous",         # 7
    "pure_ai_quick_dump",            # 8
]

RISK_TIER = {
    "hardworker": "low",
    "daily_sprinter": "low",
    "consistent_late_breakthrough": "low",
    "low_interest_quitter": "low",
    "benchmark_tweaker": "medium",
    "late_joiner_dangerous": "medium",
    "consistent_ai_leap": "high",
    "ai_dump_sincere_start": "high",
    "pure_ai_quick_dump": "high",
}

RISK_COLOR = {"low": "#1e7e34", "medium": "#b8860b", "high": "#c0392b"}

FEATURE_DESCRIPTIONS = {
    "total_submissions": "Number of completed submissions in total",
    "active_days_pct": "Percent of contest days with at least one submission",
    "submissions_per_day": "Average submissions per active day",
    "submission_avg_initial_days": "Average submissions/day in the first third of the contest",
    "avg_submissions_mid": "Average submissions/day in the middle third of the contest",
    "submissions_end": "Submissions made on the final day",
    "median_score": "Median public-leaderboard score",
    "top_score": "Best (lowest RMSE) public score achieved",
    "lowest_score": "Worst public score achieved",
    "gap_median_top": "Median score minus best score",
    "gap_lowest_top": "Worst score minus best score",
    "peak_submissions_per_day": "Most submissions made in a single day",
    "top_score_on_peak_day": "Best score achieved on that peak day",
    "peak_submissions_per_hour": "Most submissions made in a single hour",
    "top_score_on_peak_hour": "Best score achieved in that peak hour",
    "claude_min_score": "Reference: Claude's own worst score on this contest",
    "user_min_score_diff": "Participant's best score minus claude_min_score",
    "above_below_claude_min": "1 if participant's best score beat claude_min_score",
    "score_time_corr": "Correlation between submission time and score",
    "time_to_best_score_hours": "Hours from first submission to best score",
    "submissions_pre_jump": "Submissions before the largest single score jump",
    "submissions_post_jump": "Submissions after the largest single score jump",
    "leap_size": "Size of the largest single score jump",
    "post_jump_score_repetition": "How repetitive scores are after that jump (0-1)",
    "quiz_participation_pct": "Percent of summer-training quizzes attended",
    "final_rank": "Final leaderboard rank",
    "rank_to_submissions": "final_rank / total_submissions",
    "rank_to_quiz_participation": "final_rank / (quiz_participation_pct/10 + 1)",
    "description_rate": "Fraction of submissions with a written description",
}


def load_models(xgb_path="xgboost_model.joblib", dt_path="decision_tree.joblib"):
    xgb = joblib.load(xgb_path)
    dt = joblib.load(dt_path)
    return xgb, dt


def build_viz_model(dt, dataset_path="cheating_detection_dataset_6.csv"):
    """Builds the dtreeviz model, using the training dataset only as reference
    data for drawing node distributions -- it does not refit the tree."""
    import dtreeviz

    ref = pd.read_csv(dataset_path)
    ref = ref.drop(columns=["participant_id", "competition_id"])
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    X_ref = ref[FEATURE_COLUMN_ORDER]
    y_ref = ref["archetype"].map(label_map)

    return dtreeviz.model(
        dt,
        X_ref,
        y_ref,
        target_name="archetype",
        feature_names=FEATURE_COLUMN_ORDER,
        class_names=CLASS_NAMES,
    )


def run_prediction(raw_csv_bytes, xgb, dt, viz_model, **preproc_kwargs):
    caught_warnings = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        df = pd.read_csv(io.BytesIO(raw_csv_bytes))
        log = load_submission_log(df)
        row = extract_features_from_log(log, **preproc_kwargs)

        for warning in w:
            caught_warnings.append(str(warning.message))

    xgb_pred_idx = int(xgb.predict(row)[0])
    xgb_proba = xgb.predict_proba(row)[0]
    xgb_probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(xgb_proba)}
    xgb_label = CLASS_NAMES[xgb_pred_idx]

    dt_pred_idx = int(dt.predict(row)[0])
    dt_label = CLASS_NAMES[dt_pred_idx]

    explanation_svg, explanation_text = None, None
    if viz_model is not None:
        # dtreeviz's tree-walker indexes x positionally (x[t.feature()] with
        # an integer feature index). Passing a pandas Series (string-indexed,
        # e.g. row.iloc[0]) raises KeyError -- confirmed by direct testing.
        # Must pass a plain numpy array instead.
        x_vec = row.iloc[0].to_numpy()
        try:
            view = viz_model.view(x=x_vec, show_just_path=True)
            explanation_svg = view.svg()
        except Exception as exc:
            caught_warnings.append(f"Could not render dtreeviz path: {exc}")
        try:
            explanation_text = viz_model.explain_prediction_path(x_vec)
        except Exception:
            explanation_text = None

    return {
        "row": row,
        "xgboost": {
            "archetype": xgb_label,
            "risk_tier": RISK_TIER.get(xgb_label, "medium"),
            "probabilities": xgb_probabilities,
            "top_probability": xgb_probabilities[xgb_label],
        },
        "decision_tree": {
            "archetype": dt_label,
            "risk_tier": RISK_TIER.get(dt_label, "medium"),
        },
        "explanation": {"svg": explanation_svg, "text": explanation_text},
        "warnings": caught_warnings,
    }


def probability_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "white", "size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#2f6fed"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#2f6fed",
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    return fig
