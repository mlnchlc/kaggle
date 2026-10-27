# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal Kaggle tabular-competition workspace (the owner is learning ML, aiming for DL/LLMs). There is no build, lint or test suite. Work happens in Jupyter notebooks, one folder per competition: `Titanic/`, `Spaceship_Titanic/`, `Airline_Satisfaction/` (Playground S6E10), `Advanced House Prices/`. `CHANGELOG.md` records past leakage and pipeline bug fixes. Check it before "fixing" something that looks odd.

## Environment

- Use the repo venv: `.venv/bin/python` (VS Code is already pointed at it). Run a notebook headlessly with `.venv/bin/jupyter nbconvert --to notebook --execute <nb>.ipynb`. Open notebooks from their own folder, because every path is relative to the notebook.
- Kaggle API auth comes from `~/.kaggle/` (`access_token` or `kaggle.json`). `.env` holds TabPFN keys and is gitignored. Never print or commit it.
- The machine has 4 cores. Notebooks size threads to fit (for example 2 XGBoost threads × 2 CV jobs).

## Architecture

- `kaggle_utils.py` at the repo root is the only shared code: `pull_competition_data(slug, data_dir="data")` (downloads and unzips, skips if `train.csv` and `test.csv` exist), `submit_prediction(file, slug, message)` (submits and polls for the public score), `show_submissions(slug, n)`. Each competition folder has a **relative symlink** `kaggle_utils.py -> ../kaggle_utils.py`. Edit only the root copy and keep the symlinks relative. New competition folders need the same symlink.
- Each notebook follows the same flow: pull data into `data/`, feature engineering, tuning, final training, write `data/submission_*.csv`, then `submit_prediction`. `data/` and submission CSVs are gitignored.
- Competition slugs are passed to `pull_competition_data` and `submit_prediction` as string literals, for example `"playground-series-s6e10"`.
- The Airline XGBoost notebook is the most developed template:
  - `engineer_features` is a stateless row-wise function, so there is no cross-row leakage.
  - `run_cv` runs StratifiedKFold with early stopping, returns OOF predictions, and averages test predictions over the fold models.
  - Optuna tunes on a 100k-row subsample (`TUNE_SAMPLE_SIZE`) at a high learning rate. The final run uses all the data at a lower learning rate.
  - SHAP is computed on held-out fold rows only.
- `Titanic/` also contains older standalone scripts (`pipeline.py`, `train_titanic*.py`, `titanic_mine.py`). `pipeline.py` is a fragment that needs `X`, `y` and the column lists defined, so it does not run on its own. Titanic and Spaceship notebooks use one `build_pipeline(params)` for tuning, holdout and final fit.

## Conventions that matter

- **Avoid leakage.** Fit imputers and encoders on train rows only, and clone shared preprocessors with `clone(...)` per pipeline. Use `StratifiedGroupKFold` when rows come in groups (Spaceship Titanic passenger groups).
- **Final submissions refit on all training data**, or average fold models. A holdout-trained model is only for reporting scores.
- Report the metric the competition actually uses (ROC-AUC for Airline) and label it correctly.
- **No external or "original" datasets for training.** The user considers this cheating, so don't suggest or add them.
