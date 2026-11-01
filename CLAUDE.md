# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal Kaggle tabular-competition workspace (the owner is learning ML, aiming for DL/LLMs). There is no build, lint or test suite. Work happens in Jupyter notebooks in `notebooks/`, one per competition approach: Titanic, Spaceship Titanic, Airline Satisfaction (Playground S6E10) and Advanced House Prices. `scripts/` holds the same notebooks as plain `.py` files for agentic work (see Architecture). `CHANGELOG.md` records past leakage and pipeline bug fixes. Check it before "fixing" something that looks odd.

## Environment

- Use the repo venv: `.venv/bin/python` (VS Code is already pointed at it). Run a notebook headlessly with `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute <nb>.ipynb`. Open notebooks with `notebooks/` as the working directory, because every path is relative to it.
- **Packages: use `uv`, not pip** (pip is uninstalled from the venv). Dependencies live in `pyproject.toml` with the exact versions in `uv.lock` (commit both). Rebuild the venv in a fresh clone with `uv sync`. Add a package with `uv add <pkg>`, upgrade with `uv lock --upgrade && uv sync`, inspect with `uv pip list` / `uv pip check`. Do not `uv pip install` for anything you want to keep, because `uv sync` removes packages that are not in the lock. `pyproject.toml` overrides `tabpfn-client`'s pins on pandas, pyarrow and scikit-learn, and `tabpfn-client` itself is not installed. The TabPFN script has not been run on the current versions.
- Kaggle API auth comes from `~/.kaggle/` (`access_token` or `kaggle.json`). `.env` holds TabPFN keys and is gitignored. Never print or commit it.
- The machine has 4 cores. Notebooks size threads to fit (for example 2 XGBoost threads × 2 CV jobs).

## Architecture

- `lib/kaggle_utils.py` is the only shared code: `pull_competition_data(slug, data_dir="data")` (downloads and unzips, skips if `train.csv` and `test.csv` exist), `submit_prediction(file, slug, message)` (submits and polls for the public score), `show_submissions(slug, n)`, `log_run(...)` (appends one finished run to the experiment log) and `load_runs(slug)` (reads it back as a DataFrame). `notebooks/` and `scripts/` each have a **relative symlink** `kaggle_utils.py -> ../lib/kaggle_utils.py`. Edit only the copy in `lib/` and keep the symlinks relative. Any new code folder needs the same symlink.
- Each notebook follows the same flow: pull data into `../data/<slug>/`, feature engineering, tuning, final training, write `../data/<slug>/submission_*.csv`, then `submit_prediction`. All data lives in one shared `data/<slug>/` folder at the repo root (for example `data/titanic/`), used by both `notebooks/` and `scripts/`, so competitions never share a folder and nothing is downloaded twice. Notebooks and scripts run with their own folder as the working directory, hence the `..`. `data/` and submission CSVs are gitignored.
- Notebooks are named `<kaggle-slug>-<tag>.ipynb`, for example `titanic-xgb.ipynb`, `spaceship-titanic-xgb.ipynb`, `playground-series-s6e10-airline-xgb.ipynb`. Keep new files in this scheme.
- **Experiment log:** call `log_run(competition, model, metric, cv_mean, cv_std, params, ...)` at the end of every real run, after `submit_prediction` so `public_score=result.public_score` can be passed (it returns the scored submission). It appends one JSON line to `runs.jsonl` at the repo root with the metric name, CV score, public score, hyperparameters, seed, git commit and dirty flag, and library versions. Always pass the competition's real `metric`. `runs.jsonl` is gitignored (it holds leaderboard scores); read it with `load_runs("titanic")` to see what has already been tried before starting a new experiment. Every script in `scripts/` ends with its own `log_run` cell, so a new script must too. `titanic-forest` and `titanic-xgb` never submit to Kaggle, so their rows have no public score.
- **Optuna storage:** every tuning script stores its trials in `data/<slug>/optuna.db` (SQLite, gitignored) through `storage="sqlite:///" + os.path.join(data_dir, "optuna.db")`. Each run uses a fresh study named `<script-name>-<timestamp>` (`STUDY_NAME`), never `load_if_exists`, so trials from different feature sets or CV setups cannot mix. `STUDY_NAME` is saved in the run's `log_run` record as `optuna_study`; reload a study with `optuna.load_study(study_name=..., storage=...)`.
- Competition slugs are passed to `pull_competition_data` and `submit_prediction` as string literals, for example `"playground-series-s6e10"`.
- The Airline XGBoost notebook is the most developed template:
  - `engineer_features` is a stateless row-wise function, so there is no cross-row leakage.
  - `run_cv` runs StratifiedKFold with early stopping, returns OOF predictions, and averages test predictions over the fold models.
  - Optuna tunes on a 100k-row subsample (`TUNE_SAMPLE_SIZE`) at a high learning rate. The final run uses all the data at a lower learning rate.
  - SHAP is computed on held-out fold rows only.
- `scripts/` is the **source of truth**: plain `# %%` percent-format `.py` files (no outputs, no metadata header). Edit and create code here. `notebooks/*.ipynb` are generated from them with jupytext.
  - Names match the notebooks: `<kaggle-slug>-<tag>.py`.
  - Run them from inside `scripts/`: `cd scripts && ../.venv/bin/python <file>.py`. They read and write the same root `data/<slug>/` folders as the notebooks. `scripts/output/` is gitignored.
  - Regenerate notebooks with `python3 tools/regen_notebooks.py` (`scripts/<name>.py` becomes `notebooks/<name>.ipynb`). It uses `jupytext --update`, so outputs already in a local notebook survive. Never hand-edit code in a notebook: the next regeneration overwrites it. A new script needs no other setup.
- The Titanic and Spaceship notebooks use one `build_pipeline(params)` for tuning, holdout and final fit.

## Pushing (enforced by a hook)

- **Run `bash tools/setup.sh` once in every fresh clone.** It registers the `stripout` clean filter (`.gitattributes` applies it to `*.ipynb`, `tools/strip_outputs.py`) and sets `core.hooksPath` to `tools/hooks`. Both are local git config, so they do not travel with the repo.
- The filter removes all cell outputs from what git stores, so notebook outputs (which can hold the Kaggle username, local paths and data rows) are never committed. Working copies keep their outputs, and `git status` ignores output-only changes.
- `tools/hooks/pre-push` runs on every `git push`. It regenerates the notebooks from `scripts/`. If that changes anything it blocks the push: review, `git add notebooks`, commit, push again. It then scans every commit being pushed and blocks on data or cache files (csv, npy, `data/`, `.env`), the Kaggle username, key and token patterns, local `/home/ubuntu` paths, secret values from `.env` or `~/.kaggle`, and notebooks that carry outputs.
- So the push order is: edit `scripts/`, run `tools/regen_notebooks.py`, commit scripts and notebooks together, push. Do not use `--no-verify`, and never `git add -f` anything the hook blocks.

## Conventions that matter

- **Avoid leakage.** Fit imputers and encoders on train rows only, and clone shared preprocessors with `clone(...)` per pipeline. Use `StratifiedGroupKFold` when rows come in groups (Spaceship Titanic passenger groups).
- **Final submissions refit on all training data**, or average fold models. A holdout-trained model is only for reporting scores.
- Report the metric the competition actually uses (ROC-AUC for Airline) and label it correctly.
- **No external or "original" datasets for training.** The user considers this cheating, so don't suggest or add them.
