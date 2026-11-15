# %% [markdown]
# # Playground S6E11: Predicting Credit Card Fraud (XGBoost + five feature ideas)
#
# Same pipeline as `playground-series-s6e11-fraud-xgb` (baseline CV 0.94748, public 0.94785) plus five groups of
# features, all domain-motivated:
#   1. night window: is_night, hour_sin, hour_cos (the hour wraps past midnight, which depth-2 trees handle badly)
#   2. risk_flag_count: how many of the 8 warning signs are present (6 boolean flags, CVV retries, prior disputes)
#   3. channel/device/auth inconsistency flags (a PC cannot tap a card, an ATM transaction should use a PIN),
#      including `unusual_channel_device` for six hand-picked unlikely channel/device pairs
#   4. transaction rate: txn_count_last_24h / hours_since_last_txn, and amount x txn_count_last_24h
#   5. amount relative to the typical amount of its merchant category and channel (a CROSS-ROW statistic, so the
#      medians are fitted on each fold's training rows only, inside run_cv)
# Evaluation metric is ROC-AUC (confirmed on the competition's Evaluation tab).
# A data check before building showed the hourly fraud rate is flat and channel, device and auth method look
# independently generated, so ideas 1 and 3 are not expected to help. The run tests that.
#
# Flow: pull data, overview, features, 5-fold CV with early stopping, Optuna on a subsample, final CV run,
# submission, `log_run`.

# %%
import os
# Work from this file's folder even when an editor (e.g. Zed's REPL) starts the kernel in the repo root,
# so the kaggle_utils symlink and the ../data paths resolve.
if os.path.isfile(os.path.join("scripts", "kaggle_utils.py")):
    os.chdir("scripts")
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from kaggle_utils import pull_competition_data, show_submissions  # symlink to lib/kaggle_utils.py

COMPETITION = "playground-series-s6e11"
METRIC = "roc_auc"   # ASSUMPTION: confirm on the Evaluation tab. One of: "roc_auc", "average_precision", "logloss"
SEED = 42

# metric name -> (function of y_true, y_pred_proba, higher_is_better)
METRICS = {
    "roc_auc": (roc_auc_score, True),
    "average_precision": (average_precision_score, True),
    "logloss": (log_loss, False),
}
score_fn, HIGHER_IS_BETTER = METRICS[METRIC]

data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb_fe5.csv")

# The target is the column that is in train but not in test; the id is the sample submission's first column.
ID_COL = sample_df.columns[0]
target_candidates = [c for c in train_df.columns if c not in kaggle_test_df.columns]
assert len(target_candidates) == 1, f"expected one target column, found {target_candidates}"
TARGET = target_candidates[0]
assert sample_df.columns[1] == TARGET, f"submission column {sample_df.columns[1]!r} differs from target {TARGET!r}"
assert train_df[TARGET].nunique() == 2, "binary classification only; adapt the model and metric for other targets"
print(f"id={ID_COL!r} target={TARGET!r} train={train_df.shape} test={kaggle_test_df.shape}")
print("sample submission values:", sample_df[TARGET].head(3).tolist())

# %%
# Overview: dtypes, missing values, distinct counts, target balance
overview = pd.DataFrame({
    "dtype": train_df.dtypes.astype(str),
    "missing": train_df.isna().sum(),
    "missing_%": (train_df.isna().mean() * 100).round(2),
    "n_unique": train_df.nunique(),
})
print(overview.to_string())
print("\ntarget balance:\n", train_df[TARGET].value_counts(normalize=True).round(5).to_string())

# %%
# Numeric columns: summary stats + histograms
numeric_cols = [c for c in train_df.select_dtypes("number").columns if c not in (ID_COL, TARGET)]
print(train_df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())

n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows), squeeze=False)
for ax, col in zip(axes.flat, numeric_cols):
    ax.hist(train_df[col].dropna(), bins=30)
    ax.set_title(f"{col} (skew {train_df[col].skew():.1f})", fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(numeric_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Features. Strings become native pandas categoricals with categories fixed from TRAIN, so train and test agree.
# XGBoost handles NaN and categoricals itself. Add engineered features in engineer_features(): it must work
# row by row (no statistics across rows) so applying it to train and test separately cannot leak.
X = train_df.drop(columns=[TARGET, ID_COL])
X_kaggle_test = kaggle_test_df.drop(columns=[ID_COL])
y = train_df[TARGET].astype(int)

CATEGORICAL_FEATURES = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
for col in CATEGORICAL_FEATURES:
    categories = X[col].astype("category").cat.categories
    X[col] = pd.Categorical(X[col], categories=categories)
    X_kaggle_test[col] = pd.Categorical(X_kaggle_test[col], categories=categories)


BOOL_FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
              "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
IN_PERSON = ["POS", "ATM", "Contactless"]
# Channel/device pairs that make little physical sense (hand-picked by the owner). 13% of rows, fraud rate 1.74% vs 1.66%.
UNUSUAL_PAIRS = [("POS", "Windows PC"), ("Online", "POS Terminal"), ("In-App", "ATM Machine"),
                 ("ATM", "Windows PC"), ("Contactless", "ATM Machine"), ("In-App", "POS Terminal")]


def engineer_features(X):
    """Row-wise features (ideas 1 to 4). No statistics across rows, so train and test can be processed separately."""
    X = X.copy()
    hour = X["time_of_day_hour"]
    X["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    X["risk_flag_count"] = (X[BOOL_FLAGS].astype(int).sum(axis=1)
                            + (X["cvv_retry_count"] > 0).astype(int) + (X["prior_disputes"] > 0).astype(int))

    channel, device = X["channel"].astype(str), X["device_type"].astype(str)
    in_person = channel.isin(IN_PERSON)
    X["unusual_channel_device"] = pd.Series(list(zip(channel, device)), index=X.index).isin(UNUSUAL_PAIRS).astype(int)
    X["computer_in_person"] = (in_person & device.isin(["Windows PC", "Mac"])).astype(int)
    X["terminal_online"] = (channel.isin(["Online", "In-App"]) & device.isin(["POS Terminal", "ATM Machine"])).astype(int)
    X["atm_without_pin"] = ((channel == "ATM") & (X["auth_method"].astype(str) != "PIN")).astype(int)

    X["txn_rate"] = X["txn_count_last_24h"] / (X["hours_since_last_txn"] + 0.1)
    X["spend_24h"] = X["amount_usd"] * X["txn_count_last_24h"]
    return X


# Idea 5: typical amount per merchant category and per channel, fitted on TRAIN-FOLD rows only (see run_cv)
GROUP_COLS = ["merchant_category", "channel"]


def fit_group_stats(X_fit):
    return {c: X_fit.groupby(c, observed=True)["amount_usd"].median() for c in GROUP_COLS}


def add_group_features(X, stats):
    X = X.copy()
    for c, median in stats.items():
        X[f"amount_vs_{c}"] = X["amount_usd"] / X[c].astype(object).map(median).astype(float)
    return X


X = engineer_features(X)
X_kaggle_test = engineer_features(X_kaggle_test)
print(f"{X.shape[1]} features, {len(CATEGORICAL_FEATURES)} categorical")

# %%
# K-fold training with early stopping. Every training row gets one out-of-fold (OOF) prediction from a model
# that never saw it; the OOF score is the validation score, and test predictions average the fold models.
EVAL_METRIC = {"roc_auc": "auc", "average_precision": "aucpr", "logloss": "logloss"}[METRIC]


def run_cv(params, X, y, X_test=None, n_splits=5, trial=None, verbose=True):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    models, folds, fold_scores = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        model = XGBClassifier(
            **params,
            n_estimators=5000,            # upper bound; early stopping picks the real number
            early_stopping_rounds=100,
            eval_metric=EVAL_METRIC,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED,
            n_jobs=4,
        )
        # Group medians come from this fold's training rows only, then are applied to train, validation and test
        stats = fit_group_stats(X.iloc[tr_idx])
        X_tr = add_group_features(X.iloc[tr_idx], stats)
        X_va = add_group_features(X.iloc[va_idx], stats)
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_scores.append(score_fn(y.iloc[va_idx], oof[va_idx]))
        if X_test is not None:
            test_pred += model.predict_proba(add_group_features(X_test, stats))[:, 1] / n_splits
        models.append(model)
        folds.append(va_idx)
        if verbose:
            print(f"Fold {fold}: {METRIC} {fold_scores[-1]:.5f}  best_iteration {model.best_iteration}", flush=True)

        if trial is not None:  # Optuna pruning: abandon trials that are clearly worse after a few folds
            trial.report(np.mean(fold_scores), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {"oof": oof, "test_pred": test_pred, "models": models, "folds": folds,
            "fold_scores": fold_scores, "oof_score": score_fn(y, oof)}


# %%
# Hyperparameter search on a stratified subsample at a high learning rate (the expensive part is kept cheap).
TUNE_SAMPLE_SIZE = min(100_000, len(X) // 2)
TUNE_LEARNING_RATE = 0.1
X_tune, _, y_tune, _ = train_test_split(X, y, train_size=TUNE_SAMPLE_SIZE, random_state=SEED, stratify=y)


def objective(trial):
    params = {
        "learning_rate": TUNE_LEARNING_RATE,
        "max_depth": trial.suggest_int("max_depth", 2, 10),  # first run's best was 3, the old lower bound
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    return run_cv(params, X_tune, y_tune, n_splits=3, trial=trial, verbose=False)["oof_score"]


# Every trial is stored in <data_dir>/optuna.db (gitignored). One study per run, named by timestamp, so trials from
# different feature sets or CV setups never mix. Reload later: optuna.load_study(study_name=..., storage=...)
STUDY_NAME = f"{COMPETITION}-fraud-xgb-fe5-{datetime.now():%Y%m%d-%H%M%S}"
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage="sqlite:///" + os.path.join(data_dir, "optuna.db"),
    direction="maximize" if HIGHER_IS_BETTER else "minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=30)

print(f"Best CV {METRIC} (tuning sample):", study.best_value)
print("Best parameters:", study.best_params)

# %%
# Final 5-fold run on ALL training data with the best parameters and a lower learning rate.
FINAL_LEARNING_RATE = 0.05
final_params = {**study.best_params, "learning_rate": FINAL_LEARNING_RATE}

result = run_cv(final_params, X, y, X_test=X_kaggle_test, n_splits=5)
oof, test_pred = result["oof"], result["test_pred"]

print(f"\nOOF {METRIC}: {result['oof_score']:.5f}")
print(f"Fold {METRIC}: {np.mean(result['fold_scores']):.5f} +/- {np.std(result['fold_scores']):.5f}")

# Save OOF and test predictions so later models can be blended or stacked
np.save(os.path.join(data_dir, "oof_xgb_fe5.npy"), oof)
np.save(os.path.join(data_dir, "test_xgb_fe5.npy"), test_pred)

# %%
# Feature importance (gain), averaged over the fold models
importance = pd.DataFrame({
    "Feature": result["models"][0].feature_names_in_,
    "Importance": np.mean([m.feature_importances_ for m in result["models"]], axis=0),
}).sort_values("Importance", ascending=False)
print(importance.to_string(index=False))

# %%
# Submission: test predictions averaged over the 5 fold models. These are probabilities. If the Evaluation tab
# asks for hard 0/1 labels instead, threshold them here (choose the threshold on the OOF predictions).
submission = pd.DataFrame({ID_COL: kaggle_test_df[ID_COL], TARGET: test_pred})
assert list(submission.columns) == list(sample_df.columns) and len(submission) == len(sample_df)
submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")

# %%
# Auto-submit to Kaggle and retrieve the public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition=COMPETITION,
    message="XGBoost 5-fold, Optuna, + night, risk count, consistency, rate, amount vs group",
    wait_for_score=True,
    timeout=60,
)

# %%
show_submissions(competition=COMPETITION, n=10)

# %%
# Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs())
from kaggle_utils import log_run

log_run(
    COMPETITION, "xgb-fe5", metric=METRIC,
    cv_mean=np.mean(result["fold_scores"]), cv_std=np.std(result["fold_scores"]),
    params=final_params, seed=SEED, features=X.shape[1],
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    extra={"oof_score": result["oof_score"], "cv": "5-fold stratified",
           "tune_trials": len(study.trials), "optuna_study": STUDY_NAME, "tune_best": study.best_value,
           "tune_sample_size": TUNE_SAMPLE_SIZE},
)
