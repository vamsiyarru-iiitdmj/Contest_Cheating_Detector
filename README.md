---
title: Contest Cheating Detector
emoji: 🏆
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.37.0"
app_file: app.py
pinned: false
---

# Contest Cheating Detector

Classifies a Kaggle-style contest participant into one of 9 behavioral
archetypes (`hardworker`, `benchmark_tweaker`, `daily_sprinter`,
`consistent_late_breakthrough`, `low_interest_quitter`,
`consistent_ai_leap`, `ai_dump_sincere_start`, `late_joiner_dangerous`,
`pure_ai_quick_dump`) from their raw submission history, using an
XGBoost classifier (with probabilities) and a Decision Tree classifier
(with a visual dtreeviz decision-path explanation).

**This version runs as a single Streamlit process — no Docker SDK, no
FastAPI backend, no paid Space hardware required.** It deploys on the
free native "Streamlit SDK" Hugging Face Space tier.

## What changed from the two-process (Docker) version

The original design ran a FastAPI backend (`api.py`, port 8000) and a
Streamlit frontend (`app.py`, port 7860) as two processes in one Docker
container, talking over HTTP. That needs the Docker SDK. This version
merges everything into one Streamlit process:

| Old | New |
|---|---|
| `api.py` (FastAPI, loads models, runs preprocessing, calls dtreeviz, returns JSON) | Logic moved into `model_utils.py` — same steps, called as plain Python functions instead of over HTTP |
| `app.py` (Streamlit, calls `api.py` via `requests`) | `app.py` now calls `model_utils.py` directly, in-process |
| `Dockerfile` + `start.sh` (runs both processes) | **Deleted** — not needed for a single-process Streamlit Space |
| `requirements.txt` (fastapi, uvicorn, python-multipart, requests, streamlit, ...) | Backend-only deps removed; `plotly` added (probability gauge) |

**Delete `Dockerfile`, `start.sh`, and `api.py` from the repo** — they're
no longer used. Keeping the SDK as `docker` in this README's front matter
while removing the Dockerfile would break the Space, so the front matter
above is already switched to `sdk: streamlit`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI only — layout, form, results rendering. Delegates all logic to `model_utils.py`. |
| `model_utils.py` | Model loading, prediction, dtreeviz explanation, probability gauge. No Streamlit calls in it, so it's independently testable with plain Python (see Verification below). |
| `preprocessor.py` | Turns a raw `submission_history.csv` into the 29-column feature row the models were trained on. Unchanged. |
| `xgboost_model.joblib` / `decision_tree.joblib` | Trained models (from `start_weights_dt_2.ipynb`). Unchanged. |
| `cheating_detection_dataset_6.csv` | Training dataset, reused at runtime only to give dtreeviz reference data for drawing the tree explanation — it does not retrain anything. |
| `requirements.txt` | Python deps for the single Streamlit process. |
| `packages.txt` | System packages (`graphviz`, `libgraphviz-dev`) — required here, since this **is** the native Streamlit SDK setup packages.txt is meant for. |

`start_weights_dt_2.ipynb` stays local, not deployed, as before.

## Required inputs per prediction

Entered directly in the form (all required, since rank-based features
must not silently become NaN):

- The raw `submission_history.csv` (must have at least `date` and
  `publicScore` columns).
- `claude_min_score` — Claude's own worst score on this contest.
- `Quiz Attendance` (0–10) — external data, not in the submission log.
- `Rank` and `Total Participants` — needed for `final_rank`,
  `rank_to_submissions`, `rank_to_quiz_participation`. The app blocks
  submission if `Rank > Total Participants`.
- Optional (advanced expander): `competition_start`,
  `competition_duration_days` — used to correctly bucket day-based
  features. If left blank, the preprocessor falls back to using the
  person's own first submission as day 1 and shows a warning explaining
  the approximation.

## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Verification performed while building this

- **Confirmed the class-index mapping** (`CLASS_NAMES` order in
  `model_utils.py`) against the actual training notebook cell that built
  the `{archetype: int}` dict used to fit both models — not assumed.
- **Confirmed which decision tree was actually saved**: the notebook
  builds two trees (`dt`, `dt2`); only `dt2` (the balanced,
  `max_depth=9`, entropy-criterion one) is the one in
  `decision_tree.joblib` — checked directly against the `joblib.dump(...)`
  cell.
- **Found and fixed a real bug**: the old `api.py` called
  `_VIZ_MODEL.view(x=row.iloc[0], ...)`, passing a pandas Series
  (string-indexed). dtreeviz's tree-walker indexes `x` positionally
  (`x[t.feature()]` with an integer feature index), so this raised
  `KeyError` every time. Fixed by passing `row.iloc[0].to_numpy()`
  instead — verified working end-to-end, including
  `explain_prediction_path`.
- **Fixed a broken dependency pin**: `requirements.txt` previously
  capped `scikit-learn<1.9`, but the models were pickled with
  scikit-learn 1.9.0 (confirmed via the `InconsistentVersionWarning`
  raised when loading them under 1.8). The old pin would have forced
  exactly the wrong version on every fresh install. Now pinned to
  `scikit-learn==1.9.0`.
- **Ran the actual Streamlit server** (`streamlit run`, headless) and
  hit it with a request to confirm it serves without error, then drove
  it with Streamlit's `AppTest` framework to confirm the app boots with
  zero exceptions and the "no file uploaded" guard path renders its
  error message correctly instead of crashing.
- **Ran the full prediction pipeline against your real
  `my_submission_history.csv`** through the installed, pinned
  `requirements.txt` versions: preprocessing, both model predictions,
  and dtreeviz SVG + text explanation generation all succeed with zero
  warnings.

## Known, low-risk residual items (not fixed, flagged instead)

- Loading the XGBoost model prints a `UserWarning` recommending
  `Booster.save_model` instead of pickle for long-term version safety.
  It **does not** affect correctness here (predictions were verified
  correct) — it's a forward-compatibility suggestion from the XGBoost
  team, not an error.
- The training dataset enforced a hard cap of 10 submissions/day; your
  real submission history hit 11 in one day. The model has never seen
  values above 10 for that feature. Harmless for a single prediction,
  but worth knowing if you retrain later.
- dtreeviz's SVG output logs `findfont: Arial not found` on Linux (falls
  back to a default font automatically). Purely cosmetic — doesn't
  affect the explanation's correctness, and will likely also appear in
  the Space's logs; safe to ignore.

## Notes / assumptions carried over from the original design

- The green/amber/red badge coloring (`RISK_TIER` in `model_utils.py`)
  is a **display-only** heuristic grouping the 9 archetypes into
  low/medium/high risk for readability — it does not affect the model's
  prediction.
- The Decision Tree card intentionally shows no probability.
- The dtreeviz explanation shows the specific decision path for the
  submitted row (`show_just_path=True`), not the full tree.
- Upload dropzone border color (red before a file is chosen, green
  after) is a CSS hook on Streamlit's internal
  `data-testid="stFileUploaderDropzone"` element. This is a
  version-dependent implementation detail of Streamlit, not a public
  API — if a future Streamlit upgrade renames that test id, the color
  cue silently stops working but the app keeps functioning normally.
