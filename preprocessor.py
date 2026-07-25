"""
preprocess_submission_log.py

Turns a raw Kaggle-style submission-history export (one row per submission,
columns: ref, fileName, date, description, status, publicScore, privateScore)
into a single feature row matching the schema of the training dataset used
to build the archetype classifier.

USAGE
-----
    from preprocess_submission_log import extract_features_from_csv

    row = extract_features_from_csv(
        "my_submission_history.csv",
        claude_min_score=0.60,          # REQUIRED - external reference, see below
        quiz_participation=7,            # REQUIRED - external, not in the submission log
        competition_start="2026-06-28 15:08:00",  # REQUIRED - see below
        competition_duration_days=10,
        final_rank=None,                 # optional, see below
        total_participants=None,         # optional, see below
    )
    model.predict(row)   # row is a single-row DataFrame, columns already in
                          # the exact order the model was trained on

WHY SOME PARAMETERS ARE REQUIRED (this is not something this script can
compute from the submission log alone -- they are genuinely external facts):

  claude_min_score
      Claude's own worst/first RMSE submission on this specific competition.
      This is a fact about Claude's behavior on this problem, not about the
      user's submissions -- it must be supplied.

  quiz_participation
      Number of summer-training quizzes (0-10) this participant attended.
      This lives in a totally separate system from the Kaggle submission
      log and cannot be derived from it.

  competition_start
      The submission log only tells you WHEN this person submitted, not
      when the competition itself opened. Without a true start time,
      "active_days", "submission_avg_initial_days", "avg_submissions_mid",
      and "submissions_end" cannot be pinned to the right calendar days.
      If you truly don't have it, you can pass competition_start=None and
      the script will fall back to using this person's own first submission
      as day 1 -- but that WILL distort every day-bucketed feature for
      anyone who joined late, so it prints a loud warning when it does this.

  final_rank / total_participants
      Rank-based features (final_rank, rank_to_submissions,
      rank_to_quiz_participation) require knowing where this person placed
      relative to EVERYONE ELSE in the competition -- information that does
      not exist in a single person's submission log. Pass final_rank
      directly if you know it (e.g. from the leaderboard). If you don't,
      these three columns come back as NaN, with a warning explaining why,
      rather than silently guessing.

SCORE COLUMN USED: publicScore (not privateScore) -- because during a live
competition this is the only score a real-time cheat-detector could
possibly see. Using privateScore here would leak future information no
real monitoring system would have at submission time.
"""

import warnings
import numpy as np
import pandas as pd

MAX_SUBMISSIONS_PER_DAY_TRAINING_ASSUMPTION = 10  # for reference only; real logs are used as-is, not capped

FEATURE_COLUMN_ORDER = [
    "total_submissions", "active_days", "submissions_per_day", "submission_avg_initial_days",
    "avg_submissions_mid", "submissions_end",
    "median_score", "top_score", "lowest_score", "gap_median_top", "gap_lowest_top",
    "peak_submissions_per_day", "top_score_on_peak_day", "peak_submissions_per_hour", "top_score_on_peak_hour",
    "claude_min_score", "user_min_score_diff", "above_below_claude_min",
    "score_time_corr", "time_to_best_score_hours",
    "submissions_pre_jump", "submissions_post_jump", "leap_size", "post_jump_score_repetition",
    "quiz_participation", "final_rank",
    "rank_to_submissions", "rank_to_quiz_participation",
    "description_rate",
]


def _analyze_jump(scores):
    """Identical logic to the training-data generator's jump detector.
    No significant jump (>=0.05 RMSE) -> everything is 'pre-jump', leap_size=0,
    repetition=0 (neutral). Jump found but <2 submissions follow it -> repetition
    stays 0 (neutral, not maximum-suspicious) rather than a false positive."""
    n = len(scores)
    result = dict(submissions_pre_jump=n, submissions_post_jump=0,
                  leap_size=0.0, post_jump_score_repetition=0.0)
    if n < 4:
        return result
    diffs = -np.diff(scores)  # positive = improvement (RMSE decrease)
    jump_pos = int(np.argmax(diffs))
    jump_size = diffs[jump_pos]
    if jump_size < 0.05:
        return result
    jump_idx = jump_pos + 1
    pre, post = scores[:jump_idx], scores[jump_idx:]
    result["submissions_pre_jump"] = jump_idx
    result["submissions_post_jump"] = n - jump_idx
    result["leap_size"] = float(jump_size)
    if len(post) < 2:
        return result
    pre_std = np.std(pre) if len(pre) >= 2 else 0.05
    post_std = np.std(post)
    ratio = post_std / (pre_std + 0.01)
    result["post_jump_score_repetition"] = float(np.clip(1 - ratio, 0, 1))
    return result


def load_submission_log(csv_path_or_df):
    """Load and clean a raw submission-history export."""
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df)

    df.columns = [c.strip() for c in df.columns]
    required = {"date", "publicScore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Submission log is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Only keep rows that actually completed and produced a real score.
    if "status" in df.columns:
        n_before = len(df)
        df = df[df["status"].astype(str).str.contains("COMPLETE", case=False, na=False)].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            warnings.warn(f"Dropped {n_dropped} non-COMPLETE submission(s) (pending/failed/errored).")

    df = df[df["publicScore"].notna()].copy()
    if df.empty:
        raise ValueError("No completed submissions with a valid publicScore were found in this log.")

    # Dates come with and without fractional seconds in real Kaggle exports -> format='mixed'.
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)

    if "description" not in df.columns:
        df["description"] = np.nan

    return df


def extract_features_from_log(
    df,
    claude_min_score,
    quiz_participation,
    competition_start=None,
    competition_duration_days=10,
    final_rank=None,
    total_participants=None,
):
    """Core extraction logic. `df` must already be loaded via load_submission_log()."""

    if claude_min_score is None:
        raise ValueError("claude_min_score is required (Claude's own worst RMSE on this competition).")
    if quiz_participation is None:
        raise ValueError("quiz_participation is required (0-10, external to the submission log).")

    scores = df["publicScore"].to_numpy(dtype=float)
    timestamps_dt = df["date"]

    if competition_start is None:
        warnings.warn(
            "competition_start was not provided -- falling back to this person's own first "
            "submission as day 1. This WILL distort active_days / submission_avg_initial_days / "
            "avg_submissions_mid / submissions_end for anyone who joined the competition late, "
            "since 'day 1' here means 'their day 1', not the competition's actual day 1."
        )
        competition_start = timestamps_dt.iloc[0]
    else:
        competition_start = pd.to_datetime(competition_start)

    hours = (timestamps_dt - competition_start).dt.total_seconds() / 3600.0
    hours = hours.to_numpy(dtype=float)

    if (hours < 0).any():
        warnings.warn(
            "Some submissions occurred before competition_start. Clipping their time to 0 "
            "(hour 0 of the competition) rather than silently dropping them."
        )
        hours = np.maximum(hours, 0.0)

    n_sub = len(scores)
    days = np.floor(hours / 24).astype(int) + 1  # 1-indexed calendar day
    hours_of_day = np.floor(hours % 24).astype(int)
    comp_days = competition_duration_days

    if days.max() > comp_days:
        warnings.warn(
            f"Some submissions fall after day {comp_days} of the stated "
            f"{comp_days}-day competition (latest at day {int(days.max())}). "
            "They are still counted, but double-check competition_duration_days / competition_start."
        )

    active_days = len(np.unique(days))

    day_series = pd.Series(days)
    day_counts = day_series.value_counts()
    peak_submissions_per_day = int(day_counts.max())
    peak_day = day_counts.idxmax()
    top_score_on_peak_day = float(scores[days == peak_day].min())

    hourbin = days * 100 + hours_of_day
    hb_counts = pd.Series(hourbin).value_counts()
    peak_submissions_per_hour = int(hb_counts.max())
    peak_hb = hb_counts.idxmax()
    top_score_on_peak_hour = float(scores[hourbin == peak_hb].min())

    third = comp_days / 3.0
    initial_days_list = [d for d in range(1, comp_days + 1) if d <= third] or [1]
    initial_count = int(np.isin(days, initial_days_list).sum())
    submission_avg_initial_days = initial_count / len(initial_days_list)

    mid_days_list = [d for d in range(1, comp_days + 1) if third < d <= 2 * third] or list(range(1, comp_days + 1))
    mid_count = int(np.isin(days, mid_days_list).sum())
    avg_submissions_mid = mid_count / len(mid_days_list)

    submissions_end = int((days == comp_days).sum())

    median_score = float(np.median(scores))
    top_score = float(np.min(scores))
    lowest_score = float(np.max(scores))
    gap_median_top = median_score - top_score
    gap_lowest_top = lowest_score - top_score

    user_min_score_diff = top_score - claude_min_score
    above_below_claude_min = int(top_score < claude_min_score)

    if n_sub > 1 and np.std(hours) > 1e-9 and np.std(scores) > 1e-9:
        score_time_corr = float(np.corrcoef(hours, scores)[0, 1])
    else:
        score_time_corr = 0.0

    best_idx = int(np.argmin(scores))
    time_to_best_score_hours = float(hours[best_idx] - hours[0])

    jump_info = _analyze_jump(scores)

    submissions_per_day_rate = n_sub / active_days

    desc_col = df["description"].astype(str)
    has_desc = desc_col.notna() & (desc_col.str.strip() != "") & (desc_col.str.lower() != "nan")
    description_rate = float(has_desc.mean())

    if final_rank is not None:
        rank_to_submissions = final_rank / n_sub
        rank_to_quiz_participation = final_rank / (quiz_participation + 1)
    else:
        final_rank = np.nan
        rank_to_submissions = np.nan
        rank_to_quiz_participation = np.nan
        warnings.warn(
            "final_rank was not provided, so final_rank / rank_to_submissions / "
            "rank_to_quiz_participation are NaN. These require knowing this person's placement "
            "relative to the rest of the competition's leaderboard -- pass final_rank explicitly "
            "if you have it."
        )
    if total_participants is not None and final_rank is not None and not np.isnan(final_rank):
        if final_rank > total_participants:
            warnings.warn("final_rank is larger than total_participants -- check these inputs.")

    feats = dict(
        total_submissions=n_sub,
        active_days=active_days,
        submissions_per_day=submissions_per_day_rate,
        submission_avg_initial_days=submission_avg_initial_days,
        avg_submissions_mid=avg_submissions_mid,
        submissions_end=submissions_end,
        median_score=median_score,
        top_score=top_score,
        lowest_score=lowest_score,
        gap_median_top=gap_median_top,
        gap_lowest_top=gap_lowest_top,
        peak_submissions_per_day=peak_submissions_per_day,
        top_score_on_peak_day=top_score_on_peak_day,
        peak_submissions_per_hour=peak_submissions_per_hour,
        top_score_on_peak_hour=top_score_on_peak_hour,
        claude_min_score=claude_min_score,
        user_min_score_diff=user_min_score_diff,
        above_below_claude_min=above_below_claude_min,
        score_time_corr=score_time_corr,
        time_to_best_score_hours=time_to_best_score_hours,
        quiz_participation=quiz_participation,
        final_rank=final_rank,
        rank_to_submissions=rank_to_submissions,
        rank_to_quiz_participation=rank_to_quiz_participation,
        description_rate=round(description_rate, 3),
    )
    feats.update(jump_info)

    row = pd.DataFrame([feats])[FEATURE_COLUMN_ORDER]
    return row


def extract_features_from_csv(csv_path, **kwargs):
    """Convenience wrapper: load a raw submission-history CSV file and extract features."""
    df = load_submission_log(csv_path)
    return extract_features_from_log(df, **kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_submission_log.py <submission_history.csv>")
        sys.exit(1)

    # Minimal smoke-test run with placeholder external values so the script
    # can be sanity-checked end-to-end. Replace these with real values.
    row = extract_features_from_csv(
        sys.argv[1],
        claude_min_score=0.60,
        quiz_participation=5,
        competition_start=None,   # falls back to first submission + warns
        competition_duration_days=10,
        final_rank=None,
        total_participants=None,
    )
    pd.set_option("display.width", 160)
    print(row.T)