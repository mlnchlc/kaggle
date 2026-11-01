"""
Kaggle Automation Utilities: Auto-pull competition data & Auto-submit predictions.
Uses the Kaggle API (authenticated via ~/.kaggle/access_token or ~/.kaggle/kaggle.json).
"""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def get_kaggle_api():
    """Initializes and authenticates the Kaggle API client."""
    api = KaggleApi()
    api.authenticate()
    return api

def pull_competition_data(competition: str, data_dir: str = "data", force: bool = False):
    """
    Downloads and extracts competition files into data_dir if not already present.

    Args:
        competition: Kaggle competition slug (e.g. 'spaceship-titanic' or 'titanic').
        data_dir: Target directory to save data.
        force: If True, re-downloads even if files already exist.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if key data files already exist
    expected_files = ["train.csv", "test.csv"]
    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in expected_files)

    if all_exist and not force:
        print(f"Data files already exist in '{data_dir}'. Skipping download (use force=True to re-download).")
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    print(f"Pulling data for competition '{competition}' into '{data_dir}'...")
    api = get_kaggle_api()
    api.competition_download_files(competition, path=data_dir, force=force)

    zip_path = os.path.join(data_dir, f"{competition}.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print("Extraction complete, cleaned up archive.")
    else:
        print(f"Note: archive {zip_path} not found; files may have downloaded directly.")

    extracted = os.listdir(data_dir)
    print(f"Files available in '{data_dir}': {extracted}")
    return extracted


def submit_prediction(
    file_path: str,
    competition: str,
    message: str = "Automated submission",
    wait_for_score: bool = True,
    timeout: int = 60,
    poll_interval: int = 5,
):
    """
    Submits a prediction CSV to Kaggle and optionally polls for the public score.

    Args:
        file_path: Path to the submission CSV file.
        competition: Kaggle competition slug (e.g. 'spaceship-titanic').
        message: Description message for the submission.
        wait_for_score: Whether to wait and display the scored result.
        timeout: Maximum seconds to wait for scoring.
        poll_interval: Seconds between status checks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Submission file not found: {file_path}")

    api = get_kaggle_api()
    print(f"Submitting '{file_path}' to competition '{competition}'...")
    print(f"Message: {message}")

    api.competition_submit(
        file_name=file_path,
        message=message,
        competition=competition
    )
    print("Upload successful!")

    if not wait_for_score:
        return None

    print(f"Waiting for scoring (checking every {poll_interval}s, timeout={timeout}s)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(poll_interval)
        try:
            submissions = api.competition_submissions(competition)
            if not submissions:
                continue

            latest = submissions[0]
            status_str = str(latest.status)

            if "COMPLETE" in status_str:
                print(f"\n Submission Scored!")
                print(f"Public Score : {latest.public_score}")
                print(f"Status       : {status_str}")
                print(f"Date         : {latest.date}")
                print(f"File         : {latest.file_name}")
                return latest
            elif "ERROR" in status_str:
                print(f"\n Submission encountered an error: {latest.error_description}")
                return latest
            else:
                print(f"Current status: {status_str}... waiting...")
        except Exception as e:
            print(f"Polling error (will retry): {e}")

    print(f"\nTimed out waiting for score after {timeout}s. Check leaderboard or kaggle submissions manually.")
    return None


def show_submissions(competition: str, n: int = 5):
    """Displays the most recent submissions and scores for a competition."""
    api = get_kaggle_api()
    submissions = api.competition_submissions(competition)
    if not submissions:
        print(f"No submissions found for '{competition}'.")
        return []

    print(f"\n--- Recent Submissions for {competition} ---")
    for s in submissions[:n]:
        print(f"[{s.date}] File: {s.file_name:<20} Score: {str(s.public_score):<10} Status: {s.status}")
    return submissions[:n]


# --- Experiment log -------------------------------------------------------------------------------
# One JSON line per finished run in <repo root>/runs.jsonl (gitignored). Resolved from this file's real
# path, so the symlinks in notebooks/ and scripts/ all write to the same log.
RUN_LOG = Path(__file__).resolve().parent.parent / "runs.jsonl"
_TRACKED_LIBS = ("xgboost", "lightgbm", "catboost", "scikit-learn", "optuna", "pandas", "numpy")


def _json_default(obj):
    """Lets numpy scalars/arrays and other odd types in params serialize instead of crashing."""
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _to_float(value):
    """float(value), or None for None, empty or non-numeric values (e.g. a score that timed out)."""
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _git_state():
    """Short commit hash and whether the working tree has uncommitted changes (None if not a repo)."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def log_run(competition: str, model: str, metric: str, cv_mean: float, cv_std: float = None, params: dict = None,
            public_score=None, submission_file: str = None, seed: int = None, features=None, notes: str = None,
            extra: dict = None, log_file=None):
    """
    Appends one finished run to the experiment log (one JSON object per line) and returns the record.

    Args:
        competition: Kaggle competition slug, e.g. 'spaceship-titanic'.
        model: Short model name, e.g. 'xgb', 'forest', 'tabpfn'.
        metric: The competition's real metric, e.g. 'roc_auc', 'accuracy', 'rmse_log_price'.
        cv_mean / cv_std: Cross-validation score (mean and standard deviation across folds).
        params: Hyperparameters of the final model (numpy types are fine).
        public_score: Leaderboard score, e.g. from submit_prediction(...).public_score.
        submission_file: Path of the submitted CSV (only the file name is stored).
        seed: Random seed used for the final fit and the CV split.
        features: Feature names, or an int count, used by the model.
        notes: Free text, e.g. what changed in this run.
        extra: Anything else worth keeping, e.g. {"optuna_study": "airline-v3", "n_trials": 50}.
        log_file: Override the default <repo root>/runs.jsonl.
    """
    commit, dirty = _git_state()
    versions = {}
    try:
        from importlib import metadata
        for lib in _TRACKED_LIBS:
            try:
                versions[lib] = metadata.version(lib)
            except metadata.PackageNotFoundError:
                pass
    except ImportError:
        pass

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "competition": competition,
        "model": model,
        "metric": metric,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "public_score": _to_float(public_score),
        "params": params or {},
        "seed": seed,
        "features": features,
        "submission_file": os.path.basename(submission_file) if submission_file else None,
        "notes": notes,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "versions": versions,
        **(extra or {}),
    }
    path = Path(log_file) if log_file else RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    print(f"Logged run to {path}: {competition} / {model} / {metric} cv={cv_mean:.5f}"
          + (f" public={record['public_score']:.5f}" if record["public_score"] is not None else ""))
    return record


def load_runs(competition: str = None, log_file=None):
    """Returns the experiment log as a DataFrame (newest last), optionally for one competition."""
    import pandas as pd
    path = Path(log_file) if log_file else RUN_LOG
    if not path.exists():
        return pd.DataFrame()
    runs = pd.read_json(path, lines=True)
    if competition is not None and "competition" in runs:
        runs = runs[runs["competition"] == competition]
    return runs.reset_index(drop=True)
