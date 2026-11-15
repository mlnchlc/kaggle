# Kaggle tabular notebooks

My working repo for Kaggle tabular competitions. I use it to learn applied ML properly (validation, leakage control, tuning, experiment tracking) on the way to deep learning and LLMs, so the code favours clear, honest pipelines over leaderboard tricks.

## Competitions

| Competition | Approach | Metric | Best CV | Public LB |
|---|---|---|---|---|
| [Titanic](https://www.kaggle.com/competitions/titanic) | XGBoost, random forest | accuracy | 0.8315 | not submitted |
| [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) | XGBoost, group-aware CV | accuracy | 0.8126 | 0.8041 |
| [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | XGBoost | RMSE of log price | 0.1191 | 0.1214 |
| [Playground S6E10: Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10) | XGBoost, TabPFN | ROC-AUC | 0.9588 | 0.9583 |
| [Playground S6E11: Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11) | XGBoost, then a blend with LightGBM, CatBoost and logistic regression | ROC-AUC | 0.9477 | 0.9479 |

[`CHANGELOG.md`](CHANGELOG.md) records the experiments behind these numbers, including the ones that did not help, and past leakage and pipeline bugs that were fixed.

## Ground rules

- **No external or "original" datasets.** Playground competitions are generated from a public dataset, and adding it to training is common on the leaderboard. It is not used here.
- **Features must make domain sense.** No encodings that exploit how the synthetic data was generated (decimal digits, count or per-value encodings of exact numbers). Two airline scripts break this rule on purpose, to measure what such tricks are worth, and say so in their headers: `playground-series-s6e10-airline-xgb-allfeat` and `playground-series-s6e10-airline-xgb-top` (a credited port of a public notebook).
- **No leakage.** Encoders, imputers and group statistics are fitted on each fold's training rows only. Final predictions average the fold models or refit on all training data.
- **Trust CV, not the public leaderboard.** Every model uses the same folds and saves its out-of-fold predictions, so two runs can be compared with a paired test (`compare_oof`, below) instead of eyeballing the fold standard deviation.

## Layout

```
scripts/     source of truth: one percent-format (# %%) .py file per approach
notebooks/   the same files as Jupyter notebooks, generated from scripts/ with jupytext
lib/         kaggle_utils.py, the only shared code (symlinked into scripts/ and notebooks/)
tools/       notebook regeneration, the output-stripping git filter and the pre-push hook
data/        per-competition data, predictions and Optuna studies (gitignored, created on first run)
```

Files are named `<kaggle-slug>-<tag>`, for example `scripts/spaceship-titanic-xgb.py` and `notebooks/spaceship-titanic-xgb.ipynb`.

`lib/kaggle_utils.py` provides:

- `pull_competition_data(slug, data_dir)`: downloads and unzips the competition data, once.
- `submit_prediction(file, slug, message)` and `show_submissions(slug)`: submit and wait for the public score.
- `log_run(...)` and `load_runs(slug)`: an experiment log (`runs.jsonl`, gitignored) with the metric, CV and public scores, parameters, seed, git commit and library versions of every run.
- `compare_oof(y, oof_a, oof_b)`: a paired per-fold and bootstrap test of whether model B really beats model A.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account.

```bash
git clone <this repo> && cd <repo>
uv sync                 # creates .venv from pyproject.toml and uv.lock
bash tools/setup.sh     # once per clone: strip notebook outputs on commit, enable the pre-push hook
```

Kaggle API credentials go in `~/.kaggle/` (see the [Kaggle API docs](https://github.com/Kaggle/kaggle-api)). Accept each competition's rules on its Kaggle page before the first run, or the download fails with a 403. The TabPFN script also needs TabPFN API keys in a `.env` file at the repo root (gitignored).

## Running

Scripts run from inside `scripts/` and read and write `../data/<slug>/`:

```bash
cd scripts
../.venv/bin/python playground-series-s6e11-fraud-xgb.py
```

Each script pulls the data, engineers features, tunes with Optuna on a subsample, runs a final 5-fold CV, writes a submission, submits it to Kaggle and logs the run. To work interactively, open the matching notebook in `notebooks/` with `notebooks/` as the working directory.

Edit code in `scripts/` only and regenerate the notebooks with `python3 tools/regen_notebooks.py`; the pre-push hook checks they are in sync.

## What is not in the repo

- **Competition data and predictions.** Kaggle's terms do not allow redistributing the data; `pull_competition_data` downloads it.
- **Notebook outputs.** A git filter strips them on commit, because outputs can contain data rows, paths and account details. Run a notebook to see its plots and tables.
- **Secrets.** The pre-push hook blocks data files, credentials, token patterns and notebooks with outputs.
