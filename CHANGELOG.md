# Changelog

All notable changes to these Kaggle notebooks. Dates are when the change was made.

## 2026-10-31

Older entries below use the original file names and paths.

### Added
- **`scripts/`**: every notebook as a plain `# %%` percent-format `.py` file (jupytext, no outputs) for agentic work. Names match the notebooks.
- `advanced house prices` notebook (`house-prices-advanced-regression-techniques-xgb`) and `CLAUDE.md`.

### Changed
- **Flat layout.** The per-competition folders are gone. Notebooks live in `notebooks/` and are named `<kaggle-slug>-<tag>.ipynb` (`titanic-xgb`, `titanic-forest`, `spaceship-titanic-xgb`, `playground-series-s6e10-airline-xgb`, `playground-series-s6e10-airline-tabpfn`, `house-prices-advanced-regression-techniques-xgb`).
- **One shared `data/<slug>/` folder at the repo root** (gitignored) for both `notebooks/` and `scripts/`, replacing a `data/` folder per competition. Notebooks and scripts set `data_dir = ../data/<slug>` and run from their own folder.
- `kaggle_utils.py` symlinks now live in `notebooks/` and `scripts/` (relative, to the root copy).

### Added (experiment log)
- **`log_run()` and `load_runs()` in `kaggle_utils.py`.** `log_run` appends one JSON line per finished run (metric, CV mean/std, public score, params, seed, git commit and dirty flag, library versions) to `runs.jsonl` at the repo root; `load_runs` returns it as a DataFrame. `runs.jsonl` is gitignored. The four XGBoost scripts (and so their notebooks) call it as their last cell, logging each script's real metric: accuracy for Titanic and Spaceship Titanic, ROC-AUC for Airline, RMSE of log price for House Prices. `titanic-xgb` never submits to Kaggle, so its row has no public score.

### Changed (push workflow)
- **`scripts/` is now the source of truth; notebooks are generated.** `tools/regen_notebooks.py` rebuilds `notebooks/<name>.ipynb` from `scripts/<name>.py` with `jupytext --update` (local outputs survive).
- **Notebook outputs are never committed.** A `stripout` git clean filter (`.gitattributes`, `tools/strip_outputs.py`) strips them on commit, so the manual scrub before each push is gone.
- **`tools/hooks/pre-push`** regenerates the notebooks and blocks the push if they changed or if any commit being pushed contains data files, the Kaggle username, key or token patterns, local paths, secret values or notebook outputs. Run `bash tools/setup.sh` once per clone to enable it.

### Removed
- `Titanic/titanic_solution.ipynb` (an early attempt) and the older standalone scripts `pipeline.py`, `titanic_mine.py`, `train_titanic.py`, `train_titanic_ml_only.py`.

### Security
- The repo was published with a fresh single-commit history, because earlier commits contained competition data. Notebook outputs showing data rows, the Kaggle username and local paths are stripped before commit.

## 2026-10-17

### Added
- `Airline_Satisfaction/airline_satisfaction_tabpfn.ipynb`: TabPFN baseline for Playground S6E10, trained on a 10k-row stratified subsample.
- `CHANGELOG.md`.

### Fixed
- **Titanic `IsChild` was wrong for missing ages** (`titanic_rf`, `titanic_xgb`). It was computed before Age imputation, and `NaN <= 12` is `False`, so all 177 passengers without an age were flagged as adults. It now runs in its own `AddIsChild` step after the age imputer.
- **Titanic RF tuned one pipeline and trained another.** The Optuna objective imputed Age by `Title, Pclass`; the final model used `Sex, Pclass`.
- **Titanic final models trained on 80% of the data.** Submissions now use a model fit on all of `X`; a separate `eval_model` fit on `X_train` gives the holdout score.
- **Shared `preprocessor` object across pipelines** (Titanic). Fitting one pipeline re-fit the preprocessor inside another. Every pipeline now gets `clone(preprocessor)`.
- **Spaceship Titanic holdout split families apart.** `train_test_split` put 41% of holdout rows in groups that also appeared in train, leaking signal and corrupting `GroupSize`. The holdout is now one fold of `StratifiedGroupKFold` (80/20, groups kept whole). Kaggle submissions were not affected, since the test set shares no groups with train.
- **`titanic_solution` leakage.** Age, Fare and Embarked fill values were computed over train and test combined; they now come from train rows only.
- **`titanic_solution` stacking score was in-sample.** The meta-learner was scored on the same OOF predictions it was fit on (reported 84.6%). It is now scored with `cross_val_predict` (83.7%).
- **Metric labels.** ROC-AUC objectives (Titanic RF, Airline) no longer print as "accuracy".

### Changed
- **One `build_pipeline(params)` per notebook**, used for tuning, holdout evaluation and the final model, replacing three copy-pasted pipeline definitions.
- **Airline XGBoost tunes on a 100k-row stratified subsample** (`TUNE_SAMPLE_SIZE`) instead of all ~560k rows; final models still train on all data. Thread counts now match the 4-core machine (2 XGBoost threads × 2 CV jobs).
- **SHAP plots explain the holdout with `eval_model`**, which never saw those rows. The Airline notebook uses a 5k-row sample.
- **Relative paths everywhere.** Notebooks read from `data/` next to the notebook and import `kaggle_utils` from their own folder; the hardcoded `/home/ubuntu/...` paths and `sys.path` hacks are gone. Titanic notebooks now use `pull_competition_data("titanic")`.
- **`kaggle_utils.py` lives only at the repo root.** `Airline_Satisfaction/kaggle_utils.py` was a copy and is now a symlink like the others; all symlinks are relative.

### Security
- The TabPFN API key is read from `.env` via `python-dotenv` instead of being hardcoded in the notebook, and `.env` is now in `.gitignore`. The key was never committed.

### Environment
- Installed `seaborn` into `.venv` (required by `titanic_solution.ipynb`).
