# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal Kaggle tabular-competition workspace (the owner is learning ML, aiming for DL/LLMs). There is no build, lint or test suite. Work happens in Jupyter notebooks in `notebooks/`, one per competition approach: Titanic, Spaceship Titanic, Airline Satisfaction (Playground S6E10) and Advanced House Prices. `scripts/` holds the same notebooks as plain `.py` files for agentic work (see Architecture). `CHANGELOG.md` records past leakage and pipeline bug fixes. Check it before "fixing" something that looks odd.

## Environment

- Use the repo venv: `.venv/bin/python` (VS Code is already pointed at it). Run a notebook headlessly with `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute <nb>.ipynb`. Open notebooks with `notebooks/` as the working directory, because every path is relative to it.
- Kaggle API auth comes from `~/.kaggle/` (`access_token` or `kaggle.json`). `.env` holds TabPFN keys and is gitignored. Never print or commit it.
- The machine has 4 cores. Notebooks size threads to fit (for example 2 XGBoost threads × 2 CV jobs).

## Architecture

- `kaggle_utils.py` at the repo root is the only shared code: `pull_competition_data(slug, data_dir="data")` (downloads and unzips, skips if `train.csv` and `test.csv` exist), `submit_prediction(file, slug, message)` (submits and polls for the public score), `show_submissions(slug, n)`. `notebooks/` and `scripts/` each have a **relative symlink** `kaggle_utils.py -> ../kaggle_utils.py`. Edit only the root copy and keep the symlinks relative. Any new code folder needs the same symlink.
- Each notebook follows the same flow: pull data into `../data/<slug>/`, feature engineering, tuning, final training, write `../data/<slug>/submission_*.csv`, then `submit_prediction`. All data lives in one shared `data/<slug>/` folder at the repo root (for example `data/titanic/`), used by both `notebooks/` and `scripts/`, so competitions never share a folder and nothing is downloaded twice. Notebooks and scripts run with their own folder as the working directory, hence the `..`. `data/` and submission CSVs are gitignored.
- Notebooks are named `<kaggle-slug>-<tag>.ipynb`, for example `titanic-xgb.ipynb`, `spaceship-titanic-xgb.ipynb`, `playground-series-s6e10-airline-xgb.ipynb`. Keep new files in this scheme.
- Competition slugs are passed to `pull_competition_data` and `submit_prediction` as string literals, for example `"playground-series-s6e10"`.
- The Airline XGBoost notebook is the most developed template:
  - `engineer_features` is a stateless row-wise function, so there is no cross-row leakage.
  - `run_cv` runs StratifiedKFold with early stopping, returns OOF predictions, and averages test predictions over the fold models.
  - Optuna tunes on a 100k-row subsample (`TUNE_SAMPLE_SIZE`) at a high learning rate. The final run uses all the data at a lower learning rate.
  - SHAP is computed on held-out fold rows only.
- `scripts/` is the agent-friendly copy of the notebooks, converted with jupytext to `# %%` percent-format `.py` files (no outputs, no metadata header).
  - Names match the notebooks: `<kaggle-slug>-<tag>.py`.
  - Run them from inside `scripts/`: `cd scripts && ../.venv/bin/python <file>.py`. They read and write the same root `data/<slug>/` folders as the notebooks. `scripts/output/` is gitignored.
  - The notebooks are the originals. Nothing syncs the two sets, so a fix in one must be repeated in the other. To regenerate a script from its notebook: `.venv/bin/jupytext --to py:percent --opt notebook_metadata_filter=-all -o scripts/<name>.py notebooks/<name>.ipynb` (then re-apply the `data_dir` path change if it was lost).
- The Titanic and Spaceship notebooks use one `build_pipeline(params)` for tuning, holdout and final fit.

## Conventions that matter

- **Avoid leakage.** Fit imputers and encoders on train rows only, and clone shared preprocessors with `clone(...)` per pipeline. Use `StratifiedGroupKFold` when rows come in groups (Spaceship Titanic passenger groups).
- **Final submissions refit on all training data**, or average fold models. A holdout-trained model is only for reporting scores.
- Report the metric the competition actually uses (ROC-AUC for Airline) and label it correctly.
- **No external or "original" datasets for training.** The user considers this cheating, so don't suggest or add them.
