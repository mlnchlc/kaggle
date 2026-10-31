# %%
import os
import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from kaggle_utils import pull_competition_data, show_submissions  # kaggle_utils.py is a symlink to the repo-root copy



# %%
# Auto-pull competition data (skips if train.csv and test.csv already exist)
# Paths are relative to this notebook's folder (VS Code/Jupyter run notebooks from their own directory)
COMPETITION = "house-prices-advanced-regression-techniques"

data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb.csv")
# <data_dir>/data_description.txt explains every column and its codes
print(train_df.shape, kaggle_test_df.shape)



# %% [markdown]
# ## EDA
# Column types, then distributions: numeric columns first, then categorical.
#

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
numeric_cols = train_df.select_dtypes("number").columns.drop("Id").tolist()
print(f"{len(numeric_cols)} numeric columns")
print(train_df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

n_cols = 6
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

n_cols = 6
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
X = train_df.drop(columns=["SalePrice", "Id"])
X_kaggle_test = kaggle_test_df.drop(columns=["Id"])

# The competition scores RMSE between log(predicted) and log(actual) price, so an error of 10%
# costs the same on a cheap house as on an expensive one. Training on log1p(SalePrice) with
# plain RMSE optimises exactly that metric; predictions are converted back with expm1.
y = np.log1p(train_df["SalePrice"])

# 2. Native categoricals: every text column becomes a pandas 'category'.
# Categories are fixed from TRAIN so train and test use identical codes
# (a test value never seen in train would become NaN).
CATEGORICAL_FEATURES = X.select_dtypes("object").columns.tolist()

for col in CATEGORICAL_FEATURES:
    categories = X[col].astype("category").cat.categories
    X[col] = pd.Categorical(X[col], categories=categories)
    X_kaggle_test[col] = pd.Categorical(X_kaggle_test[col], categories=categories)

# No imputation: XGBoost handles NaN natively and learns which branch missing values should take.
print(X.dtypes.astype(str).value_counts())  # astype(str): each category dtype is otherwise counted separately
print(f"\nColumns with missing values: {X.isna().any().sum()} train, {X_kaggle_test.isna().any().sum()} test")


# %%
# 3. Feature engineering.
# Every feature here must be computed row-by-row (no statistics across rows),
# so applying it to train and test separately can't leak anything.
# Each group was kept only if it lowered OOF RMSE (5 KFold seeds, fixed params):
#   ordinal + size: -0.0021 vs no features. Tried and dropped: house age / years since remodel,
#   OverallQual x area interactions, area ratios, MSSubClass as a category (each no better, or worse).

# Quality columns are ordered grades (Po < Fa < TA < Gd < Ex), but as categoricals XGBoost
# treats them as unordered labels. Mapping them to numbers lets one split mean "Good or better".
# NaN means the house has no such feature (no basement, no pool, ...) per data_description.txt -> 0.
QUALITY = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{col: QUALITY for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]},
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
}

def engineer_features(X):
    X = X.copy()

    # 1. Ordinal grades -> numbers (replaces the categorical column; keeping both scored worse)
    for col, mapping in ORDINAL_MAPS.items():
        X[col] = X[col].astype(object).map(mapping).fillna(0)

    # 2. Size totals: price depends on total space, which is spread over several columns
    # (fillna(0): a few test rows have no basement recorded as NaN instead of 0)
    X["TotalSF"] = X["TotalBsmtSF"].fillna(0) + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalBath"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                      + X["BsmtFullBath"].fillna(0) + 0.5 * X["BsmtHalfBath"].fillna(0))
    X["TotalPorchSF"] = X[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].sum(axis=1)

    return X

X = engineer_features(X)
X_kaggle_test = engineer_features(X_kaggle_test)
print(f"{X.shape[1]} features")


# %%
# 4. K-fold training with early stopping
# Each fold trains on 4/5 of the data and predicts the remaining 1/5, so every training
# row gets exactly one out-of-fold (OOF) prediction from a model that never saw it.
# - OOF RMSE over all rows is the validation score (no separate holdout needed)
# - early stopping on the fold's validation set picks the number of trees
# - test predictions are averaged over the fold models (no refit needed)
# Plain KFold rather than StratifiedKFold: stratifying needs classes, and the target is continuous.
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_cv(params, X, y, X_test=None, n_splits=5, trial=None, verbose=True):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    models, folds, fold_scores = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        model = XGBRegressor(
            **params,
            n_estimators=10000,           # upper bound; early stopping picks the real number
            early_stopping_rounds=200,    # stop after 200 rounds without validation improvement
            eval_metric="rmse",
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

        # predict automatically uses the best iteration found by early stopping
        oof[va_idx] = model.predict(X.iloc[va_idx])
        fold_scores.append(rmse(y.iloc[va_idx], oof[va_idx]))
        if X_test is not None:
            test_pred += model.predict(X_test) / n_splits
        models.append(model)
        folds.append(va_idx)

        if verbose:
            print(f"Fold {fold}: RMSE {fold_scores[-1]:.5f}  best_iteration {model.best_iteration}")

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
        "oof_rmse": rmse(y, oof),
    }



# %%
# 5. Hyperparameter search
# The dataset is small (1,460 rows), so tuning uses all rows and all 5 folds - a smaller sample
# would make the score too noisy to compare trials. Tuning uses the same learning rate as the final run, so the chosen parameters match the final fit.
# n_estimators is not tuned - early stopping chooses it.
TUNE_LEARNING_RATE = 0.01

def objective(trial):
    params = {
        "learning_rate": TUNE_LEARNING_RATE,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 0.1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    result = run_cv(params, X, y, n_splits=5, trial=trial, verbose=False)
    return result["oof_rmse"]


# Seeded sampler -> re-running the notebook reproduces the same search.
# direction="minimize": lower RMSE is better.
from datetime import datetime

# Every trial is stored in <data_dir>/optuna.db (gitignored). One study per run, named by timestamp, so trials
# from different feature sets or CV setups never mix. Reload later: optuna.load_study(study_name=..., storage=...)
STUDY_NAME = f"house-prices-advanced-regression-techniques-xgb-{datetime.now():%Y%m%d-%H%M%S}"
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage="sqlite:///" + os.path.join(data_dir, "optuna.db"),
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=50)

print("Best CV RMSE:", study.best_value)
print("Best parameters:", study.best_params)



# %%
# 6. Final 5-fold run with the best parameters.
# Same learning rate as tuning (0.01), so the tuned parameters are used exactly as they were selected.
# Early stopping picks the number of trees per fold.
FINAL_LEARNING_RATE = TUNE_LEARNING_RATE
final_params = {**study.best_params, "learning_rate": FINAL_LEARNING_RATE}

result = run_cv(final_params, X, y, X_test=X_kaggle_test, n_splits=5)
oof, test_pred = result["oof"], result["test_pred"]

print(f"\nOOF RMSE (log price, the competition metric): {result['oof_rmse']:.5f}")
print(f"Fold RMSE: {np.mean(result['fold_scores']):.5f} +/- {np.std(result['fold_scores']):.5f}")

# Save OOF and test predictions (log scale) so later models can be blended
np.save(os.path.join(data_dir, "oof_xgb.npy"), oof)
np.save(os.path.join(data_dir, "test_xgb.npy"), test_pred)



# %%
# Errors in dollars, for intuition only (the competition scores RMSE on log prices)
price_true, price_pred = np.expm1(y), np.expm1(oof)
print(f"Mean absolute error: ${mean_absolute_error(price_true, price_pred):,.0f}")
print(f"Median absolute % error: {np.median(np.abs(price_pred / price_true - 1)):.1%}")

# The houses the model gets most wrong are worth a look
errors = pd.DataFrame({
    "Id": train_df["Id"],
    "SalePrice": price_true.round(),
    "Predicted": price_pred.round(),
    "LogError": oof - y,
})
print(errors.reindex(errors["LogError"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))



# %%
# Feature importance (gain), averaged over the fold models
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.mean([m.feature_importances_ for m in result["models"]], axis=0),
}).sort_values("Importance", ascending=False)

print(importance.head(25).to_string(index=False))



# %%
# SHAP values for rows the model did NOT train on: fold 0's model on fold 0's validation rows.
# XGBoost computes exact SHAP values natively (pred_contribs=True), which also supports categorical splits.
# Values are in log-price units: +0.10 means the feature pushed the prediction up by about 10%.
model = result["models"][0]
X_shap = X.iloc[result["folds"][0]]

dmatrix = xgb.DMatrix(X_shap, enable_categorical=True)
contribs = model.get_booster().predict(
    dmatrix, pred_contribs=True, iteration_range=(0, model.best_iteration + 1)
)
shap_values = contribs[:, :-1]  # last column is the bias term

# Summary plot colours points by feature value, so give it category codes instead of strings
X_shap_numeric = X_shap.apply(lambda c: c.cat.codes if c.dtype == "category" else c)
shap.summary_plot(shap_values, X_shap_numeric, feature_names=list(X.columns))



# %%
# Submission: test predictions averaged over the 5 fold models, converted back from log to dollars
submission = pd.DataFrame({
    "Id": kaggle_test_df["Id"],
    "SalePrice": np.expm1(test_pred),
})

submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")



# %%
# 7. Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition=COMPETITION,
    message="XGBoost 5-fold with early stopping, native categoricals, Optuna tuning",
    wait_for_score=True,
    timeout=60
)



# %%
show_submissions(competition=COMPETITION, n=10)

# %%
# 8. Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs())
from kaggle_utils import log_run

log_run(
    COMPETITION, "xgb", metric="rmse_log_price",   # lower is better
    cv_mean=np.mean(result["fold_scores"]), cv_std=np.std(result["fold_scores"]),
    params=final_params, seed=42, features=X.shape[1],
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    extra={"oof_rmse": result["oof_rmse"], "cv": "5-fold KFold",
           "tune_trials": len(study.trials), "optuna_study": STUDY_NAME, "tune_best_rmse": study.best_value},
)
