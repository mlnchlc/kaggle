# %% [markdown]
# # Playground S6E11: Predicting Credit Card Fraud (blend of XGBoost, LightGBM, CatBoost, logistic regression)
#
# A learning exercise in blending. Four different models are trained on the SAME 5 folds, so their out-of-fold (OOF)
# predictions line up row by row. Then the notebook shows:
#   1. how correlated the models' errors are (the reason a blend helps or does not),
#   2. four ways to combine them: mean, rank mean, fitted weights, stacking,
#   3. an honest score for the fitted combinations (weights fitted on one half of the OOF rows, scored on the other).
#
# Features are the raw columns only (no engineering). The XGBoost parameters are the best ones from the tuned baseline
# `playground-series-s6e11-fraud-xgb` (CV 0.94748). The other models use sensible fixed parameters and are NOT tuned,
# so a gap between models says little about the algorithms. Metric ROC-AUC is still an ASSUMPTION (see baseline).
#
# Set SMOKE=1 in the environment for a tiny, fast check run (subsample, few trees, no submission).

# %%
import os
# Work from this file's folder even when an editor (e.g. Zed's REPL) starts the kernel in the repo root,
# so the kaggle_utils symlink and the ../data paths resolve.
if os.path.isfile(os.path.join("scripts", "kaggle_utils.py")):
    os.chdir("scripts")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from kaggle_utils import pull_competition_data, show_submissions  # symlink to lib/kaggle_utils.py

COMPETITION = "playground-series-s6e11"
METRIC = "roc_auc"   # ASSUMPTION, see the baseline script
SEED = 42
SMOKE = bool(os.environ.get("SMOKE"))
N_SPLITS = 5
MAX_TREES = 60 if SMOKE else 5000

data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_blend.csv")

ID_COL = sample_df.columns[0]
TARGET = next(c for c in train_df.columns if c not in kaggle_test_df.columns)
if SMOKE:
    train_df = train_df.sample(30_000, random_state=SEED).reset_index(drop=True)

X = train_df.drop(columns=[TARGET, ID_COL])
X_test = kaggle_test_df.drop(columns=[ID_COL])
y = train_df[TARGET].astype(int)
CAT_COLS = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
BOOL_COLS = X.select_dtypes(include="bool").columns.tolist()
NUM_COLS = [c for c in X.columns if c not in CAT_COLS and c not in BOOL_COLS]
print(f"train={X.shape} test={X_test.shape} categorical={len(CAT_COLS)} target={TARGET!r}")


def as_category(df):
    """Native pandas categoricals with categories fixed from TRAIN, so train and test agree (XGBoost, LightGBM)."""
    df = df.copy()
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=X[col].astype("category").cat.categories)
    return df


def as_strings(df):
    """CatBoost wants plain string columns for its categorical features."""
    df = df.copy()
    df[CAT_COLS] = df[CAT_COLS].astype(str)
    return df


X_cat, X_test_cat = as_category(X), as_category(X_test)
X_str, X_test_str = as_strings(X), as_strings(X_test)

# %%
# One set of folds shared by every model. This is what lets the OOF predictions be compared and combined.
folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

# Best parameters from the tuned XGBoost baseline (run with CV 0.94748)
XGB_PARAMS = {"max_depth": 2, "min_child_weight": 2.88277015446207, "subsample": 0.6607732641409216,
              "colsample_bytree": 0.6657998619189928, "gamma": 1.9186280096845645, "reg_alpha": 2.5764954897191546e-08,
              "reg_lambda": 0.004712362441040134, "learning_rate": 0.05}


def make_xgb():
    return XGBClassifier(**XGB_PARAMS, n_estimators=MAX_TREES, early_stopping_rounds=100, eval_metric="auc",
                         tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=4)


def make_lgbm():
    # num_leaves=4 is roughly the capacity of depth-2 trees, which suited this data
    return LGBMClassifier(n_estimators=MAX_TREES, learning_rate=0.05, num_leaves=4, min_child_samples=50,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                          random_state=SEED, n_jobs=4, verbose=-1)


def make_cat():
    return CatBoostClassifier(iterations=MAX_TREES, learning_rate=0.1, depth=4, eval_metric="AUC",
                              early_stopping_rounds=100, cat_features=CAT_COLS, random_seed=SEED,
                              thread_count=4, verbose=False)


def make_logreg():
    # A linear model needs scaled numbers and one-hot categories. Skewed money columns are log-transformed first.
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS + BOOL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=300))


def prep_linear(df):
    df = df.copy()
    for col in ("amount_usd", "account_balance_usd"):
        df[col] = np.log1p(df[col])
    df[BOOL_COLS] = df[BOOL_COLS].astype(int)
    return df


X_lin, X_test_lin = prep_linear(X), prep_linear(X_test)

# name -> (builder, train frame, test frame, fit kwargs builder)
MODELS = {
    "xgb": (make_xgb, X_cat, X_test_cat, lambda m, Xv, yv: {"eval_set": [(Xv, yv)], "verbose": False}),
    "lgbm": (make_lgbm, X_cat, X_test_cat,
             lambda m, Xv, yv: {"eval_set": [(Xv, yv)], "eval_metric": "auc", "callbacks": [early_stopping(100, verbose=False)]}),
    "cat": (make_cat, X_str, X_test_str, lambda m, Xv, yv: {"eval_set": (Xv, yv)}),
    "logreg": (make_logreg, X_lin, X_test_lin, lambda m, Xv, yv: {}),
}

# %%
# Train every model on the same folds. Early stopping uses the validation fold, as in the other scripts, so each
# OOF score is very slightly optimistic. It is the same small bias for every model.
oof = {name: np.zeros(len(X)) for name in MODELS}
test_pred = {name: np.zeros(len(X_test)) for name in MODELS}
fold_scores = {name: [] for name in MODELS}

for name, (build, Xtr_all, Xte, fit_kwargs) in MODELS.items():
    for fold, (tr, va) in enumerate(folds):
        model = build()
        model.fit(Xtr_all.iloc[tr], y.iloc[tr], **fit_kwargs(model, Xtr_all.iloc[va], y.iloc[va]))
        oof[name][va] = model.predict_proba(Xtr_all.iloc[va])[:, 1]
        test_pred[name] += model.predict_proba(Xte)[:, 1] / N_SPLITS
        fold_scores[name].append(roc_auc_score(y.iloc[va], oof[name][va]))
    print(f"{name:7s} OOF {METRIC} {roc_auc_score(y, oof[name]):.5f}  "
          f"folds {np.mean(fold_scores[name]):.5f} +/- {np.std(fold_scores[name]):.5f}", flush=True)

# %%
# Lesson 1: how different are the models? Rank correlation of OOF predictions (ranking is all ROC-AUC cares about).
# Blending gains come from the models disagreeing. Correlations near 1.0 mean a blend can add almost nothing.
ranks = {name: rankdata(p) / len(p) for name, p in oof.items()}
corr = pd.DataFrame(ranks).corr(method="pearson")
print("rank correlation of OOF predictions:\n", corr.round(4).to_string())

# %%
# Lesson 2: four ways to combine. All scores below are OOF ROC-AUC on the full training set.
names = list(MODELS)
R = np.column_stack([ranks[n] for n in names])            # OOF ranks, one column per model
P = np.column_stack([oof[n] for n in names])              # OOF probabilities


def auc_of(weights, M, target):
    return roc_auc_score(target, M @ weights)


def fit_weights(M, target):
    """Non-negative weights summing to 1 that maximise AUC. softmax keeps the constraints, Nelder-Mead copes with
    AUC not being smooth."""
    def loss(z):
        w = np.exp(z - z.max())
        return -auc_of(w / w.sum(), M, target)
    z = minimize(loss, np.zeros(M.shape[1]), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}).x
    w = np.exp(z - z.max())
    return w / w.sum()


results = {"best single model": max(roc_auc_score(y, oof[n]) for n in names),
           "mean of probabilities": roc_auc_score(y, P.mean(axis=1)),
           "rank mean": roc_auc_score(y, R.mean(axis=1))}

w_all = fit_weights(R, y.values)
results["fitted weights (in-sample, optimistic)"] = auc_of(w_all, R, y.values)

# Honest check: fit weights on one half of the rows, score on the other half, and the reverse. Fitting and scoring
# on the same rows can only look better than reality.
rng = np.random.default_rng(SEED)
half = rng.permutation(len(y)) < len(y) // 2
honest = []
for fit_mask in (half, ~half):
    w = fit_weights(R[fit_mask], y.values[fit_mask])
    honest.append(auc_of(w, R[~fit_mask], y.values[~fit_mask]))
results["fitted weights (cross-half, honest)"] = float(np.mean(honest))

# Stacking: a logistic regression learns the combination from the OOF predictions, itself scored out-of-fold
stack_oof = np.zeros(len(y))
for tr, va in StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + 1).split(R, y):
    meta = LogisticRegression(C=1.0, max_iter=300).fit(R[tr], y.iloc[tr])
    stack_oof[va] = meta.predict_proba(R[va])[:, 1]
results["stacking (logistic meta-model, out-of-fold)"] = roc_auc_score(y, stack_oof)

print(pd.Series(results).round(5).to_string())
print("\nfitted weights:", {n: round(float(w), 3) for n, w in zip(names, w_all)})
print("A gain smaller than the fold std "
      f"({np.std(fold_scores['xgb']):.4f}) is noise, not an improvement.")

# %%
# Pick the combination for submission: the honest fitted-weights blend if it beats the rank mean, else the rank mean.
# Test predictions are turned into ranks over the test set so they are on the same 0-1 scale as the OOF ranks.
test_ranks = np.column_stack([rankdata(test_pred[n]) / len(X_test) for n in names])
# Fitted weights must beat the rank mean by a margin on the honest score, otherwise the simple average wins.
use_weights = results["fitted weights (cross-half, honest)"] > results["rank mean"] + 0.0003
final_w = w_all if use_weights else np.full(len(names), 1 / len(names))
blend_oof = R @ final_w
blend_test = test_ranks @ final_w
BLEND_NAME = "fitted-weights rank blend" if use_weights else "rank mean"

blend_fold_scores = [roc_auc_score(y.iloc[va], blend_oof[va]) for _, va in folds]
print(f"chosen: {BLEND_NAME}, weights {({n: round(float(w), 3) for n, w in zip(names, final_w)})}")
print(f"blend OOF {METRIC}: {roc_auc_score(y, blend_oof):.5f}   "
      f"folds {np.mean(blend_fold_scores):.5f} +/- {np.std(blend_fold_scores):.5f}")

np.save(os.path.join(data_dir, "oof_blend.npy"), blend_oof)
np.save(os.path.join(data_dir, "test_blend.npy"), blend_test)

submission = pd.DataFrame({ID_COL: kaggle_test_df[ID_COL], TARGET: blend_test})
assert list(submission.columns) == list(sample_df.columns) and len(submission) == len(sample_df)
if not SMOKE:
    submission.to_csv(kaggle_output_file, index=False)
    print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")

# %%
# Auto-submit to Kaggle and retrieve the public score (skipped in the SMOKE check)
submission_result = None
if not SMOKE:
    from kaggle_utils import submit_prediction

    submission_result = submit_prediction(
        file_path=kaggle_output_file,
        competition=COMPETITION,
        message=f"Blend of XGBoost, LightGBM, CatBoost, LogReg ({BLEND_NAME})",
        wait_for_score=True,
        timeout=60,
    )
    show_submissions(competition=COMPETITION, n=10)

# %%
# Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs())
from kaggle_utils import log_run

if not SMOKE:
    log_run(
        COMPETITION, "blend", metric=METRIC,
        cv_mean=np.mean(blend_fold_scores), cv_std=np.std(blend_fold_scores),
        params={"members": names, "weights": dict(zip(names, final_w.round(4).tolist())), "method": BLEND_NAME},
        seed=SEED, features=X.shape[1],
        public_score=getattr(submission_result, "public_score", None),
        submission_file=kaggle_output_file,
        extra={"oof_score": roc_auc_score(y, blend_oof),
               "member_oof": {n: roc_auc_score(y, oof[n]) for n in names},
               "blend_results": {k: float(v) for k, v in results.items()}},
    )
