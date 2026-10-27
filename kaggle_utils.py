"""
Kaggle Automation Utilities: Auto-pull competition data & Auto-submit predictions.
Uses the Kaggle API (authenticated via ~/.kaggle/access_token or ~/.kaggle/kaggle.json).
"""

import os
import sys
import time
import zipfile
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
