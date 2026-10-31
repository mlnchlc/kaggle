# %%
import os
import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from kaggle_utils import pull_competition_data, show_submissions  # kaggle_utils.py is a symlink to the repo-root copy


# %%


# Auto-pull competition data (skips if train.csv and test.csv already exist)

# Paths are relative to this notebook's folder (VS Code/Jupyter run notebooks from their own directory)

data_dir = os.path.join("..", "data", "playground-series-s6e10")
pull_competition_data("playground-series-s6e10", data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
submission_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb.csv")
train_df.info()


# %%
import matplotlib.pyplot as plt

# Column overview: dtype, missing count and number of distinct values (train)
overview = pd.DataFrame({
    "dtype": train_df.dtypes.astype(str),
    "missing": train_df.isna().sum(),
    "missing_%": (train_df.isna().mean() * 100).round(1),
    "n_unique": train_df.nunique(),
})
print(overview["dtype"].value_counts().to_string(), "\n")
print(overview[overview["missing"] > 0].sort_values("missing", ascending=False).to_string())


# %%
# Numeric columns: summary stats + histograms (SalePrice included; skew shown in each title)
numeric_cols = train_df.select_dtypes("number").columns.drop("id").tolist()
print(f"{len(numeric_cols)} numeric columns")
print(train_df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows))
for ax, col in zip(axes.flat, numeric_cols):
    ax.hist(train_df[col].dropna(), bins=30)
    ax.set_title(f"{col} (skew {train_df[col].skew():.1f})", fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(numeric_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# Categorical columns: value counts as bar charts (missing shown as its own bar)
categorical_cols = train_df.select_dtypes("object").columns.tolist()
print(f"{len(categorical_cols)} categorical columns")

n_cols = 3
n_rows = int(np.ceil(len(categorical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.6 * n_rows))
for ax, col in zip(axes.flat, categorical_cols):
    counts = train_df[col].fillna("(missing)").value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"{col} ({len(counts)} levels)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
for ax in axes.flat[len(categorical_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# 1. Separate features and target
X = train_df.drop(columns=["satisfaction", "id"])
X_kaggle_test = kaggle_test_df.drop(columns=["id"])
y = train_df["satisfaction"].astype(int)

# 2. Native categoricals instead of OneHotEncoder.
# XGBoost (enable_categorical=True) splits on pandas 'category' columns directly.
# Categories are fixed from TRAIN so train and test use identical codes.

CATEGORICAL_FEATURES = ["Gender", "Customer Type", "Type of Travel", "Class"]

for col in CATEGORICAL_FEATURES:
    categories = X[col].astype("category").cat.categories
    X[col] = pd.Categorical(X[col], categories=categories)
    X_kaggle_test[col] = pd.Categorical(X_kaggle_test[col], categories=categories)

# No imputation: XGBoost handles NaN natively and learns which branch missing values
# should take (only 'Arrival Delay in Minutes' has NaNs, ~0.04% of rows).
print(X.dtypes.value_counts())


# %%
# 3. Feature engineering - a plain function now that there's no sklearn Pipeline.
# Every feature here must be computed row-by-row (no statistics across rows),
# so applying it to train and test separately can't leak anything.
#
# Currently no engineered features: the 21 raw columns score best. Change in OOF ROC-AUC vs raw only
# (200k-row sample, 5-fold, lr 0.1, two different samples):
#   delay (TotalDelay, LogDelay, Has*/BothDelayed, DelayRatio)           +0.00001 / -0.00004
#   survey stats over all 13 ratings (Total, Mean, Std, ZeroRatingCount) -0.00032 / -0.00023
#   comfort / ease / promptness group stats                              about -0.0003
#   Class|Type of Travel|Customer Type segment, rating counts,
#   delay catch-up / per mile, ratings as categories                     within +/-0.0003 (noise)
# With 700k rows and few, clean columns, XGBoost already learns these combinations itself.
def engineer_features(X):
    X = X.copy()

    return X

X = engineer_features(X)
X_kaggle_test = engineer_features(X_kaggle_test)
print(f"{X.shape[1]} features")


# %%
# 4. K-fold training with early stopping
# Each fold trains on 4/5 of the data and predicts the remaining 1/5, so every training
# row gets exactly one out-of-fold (OOF) prediction from a model that never saw it.
# - OOF ROC-AUC over all rows is the validation score (no separate holdout needed)
# - early stopping on the fold's validation set picks the number of trees
# - test predictions are averaged over the fold models (no refit needed)
def run_cv(params, X, y, X_test=None, n_splits=5, trial=None, verbose=True):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    models, folds, fold_scores = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        model = XGBClassifier(
            **params,
            n_estimators=5000,            # upper bound; early stopping picks the real number
            early_stopping_rounds=100,    # stop after 100 rounds without validation improvement
            eval_metric="auc",
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=4,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            verbose=False,
        )
        
        # predict_proba automatically uses the best iteration found by early stopping
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_scores.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
        if X_test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits
        models.append(model)
        folds.append(va_idx)

        if verbose:
            print(f"Fold {fold}: ROC-AUC {fold_scores[-1]:.5f}  best_iteration {model.best_iteration}")

        # Optuna pruning: abandon trials that are clearly worse than earlier ones after a few folds
        if trial is not None:
            trial.report(np.mean(fold_scores), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "oof": oof,
        "test_pred": test_pred,
        "models": models,
        "folds": folds,
        "fold_scores": fold_scores,
        "oof_auc": roc_auc_score(y, oof),
    }



# %%
# 5. Hyperparameter search
# Tuning is the expensive part, so it uses a cheaper setup than the final run:
# a 100k-row stratified sample, 3 folds and a fixed learning rate of 0.1 (fewer trees needed).
# n_estimators is no longer tuned - early stopping chooses it.

TUNE_SAMPLE_SIZE = 100_000
TUNE_LEARNING_RATE = 0.1
X_tune, _, y_tune, _ = train_test_split(
    X, y, train_size=TUNE_SAMPLE_SIZE, random_state=42, stratify=y
)

def objective(trial):
    params = {
        "learning_rate": TUNE_LEARNING_RATE,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    result = run_cv(params, X_tune, y_tune, n_splits=3, trial=trial, verbose=False)
    return result["oof_auc"]


# Seeded sampler -> re-running the notebook reproduces the same search
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=30)

print("Best CV ROC-AUC (tuning sample):", study.best_value)
print("Best parameters:", study.best_params)


# %%
# 6. Final 5-fold run on ALL training data with the best parameters.
# A lower learning rate than during tuning (more, smaller steps) usually adds a little accuracy;
# early stopping compensates by growing more trees.

FINAL_LEARNING_RATE = 0.05
final_params = {**study.best_params, "learning_rate": FINAL_LEARNING_RATE}

result = run_cv(final_params, X, y, X_test=X_kaggle_test, n_splits=5)
oof, test_pred = result["oof"], result["test_pred"]

print(f"\nOOF ROC-AUC: {result['oof_auc']:.5f}")
print(f"Fold ROC-AUC: {np.mean(result['fold_scores']):.5f} +/- {np.std(result['fold_scores']):.5f}")

# Save OOF and test predictions so later models (LightGBM, CatBoost, TabPFN) can be blended
np.save(os.path.join(data_dir, "oof_xgb.npy"), oof)
np.save(os.path.join(data_dir, "test_xgb.npy"), test_pred)


# %%
# Threshold-based metrics at 0.5, for intuition only (the competition scores ROC-AUC)
print(classification_report(y, (oof > 0.5).astype(int)))


# %%
# Feature importance (gain), averaged over the fold models

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.mean([m.feature_importances_ for m in result["models"]], axis=0),
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))


# %%
# SHAP values for rows the model did NOT train on: fold 0's model on a sample of fold 0's validation rows.
# XGBoost computes exact SHAP values natively (pred_contribs=True), which also supports categorical splits.
model = result["models"][0]
rng = np.random.default_rng(42)
shap_idx = rng.choice(result["folds"][0], size=5000, replace=False)
X_shap = X.iloc[shap_idx]

dmatrix = xgb.DMatrix(X_shap, enable_categorical=True)
contribs = model.get_booster().predict(
    dmatrix, pred_contribs=True, iteration_range=(0, model.best_iteration + 1)
)
shap_values = contribs[:, :-1]  # last column is the bias term

# Summary plot colours points by feature value, so give it category codes instead of strings
X_shap_numeric = X_shap.apply(lambda c: c.cat.codes if c.dtype == "category" else c)
shap.summary_plot(shap_values, X_shap_numeric, feature_names=list(X.columns))


# %%
# Submission: test predictions averaged over the 5 fold models
submission = pd.DataFrame({
    "id": kaggle_test_df["id"],
    "satisfaction": test_pred,   # probabilities, since the competition is scored on ROC-AUC
})

submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")


# %%
# 7. Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submit_prediction(
    file_path=kaggle_output_file,
    competition="playground-series-s6e10",
    message="XGBoost 5-fold with early stopping, native categoricals, Optuna tuning",
    wait_for_score=True,
    timeout=60
)

# %%
show_submissions(competition="playground-series-s6e10", n = 10)
