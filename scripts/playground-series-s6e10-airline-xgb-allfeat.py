# %% [markdown]
# # Airline satisfaction: XGBoost with every feature from the public "Single XGBoost Model" notebook
#
# EXPERIMENT. This deliberately includes features that break the repo's "logical features only" rule:
# frequency encoding counted over train + test, route IDs / counts / profiles built from Flight Distance,
# and digit features. It exists to measure how much of the public score they explain, next to the
# clean model in `playground-series-s6e10-airline-xgb`. Do not treat it as the model to ship.
#
# Differences from the public notebook: CPU (`device` cpu) and a learning rate of 0.05 so a full run finishes in
# reasonable time (the original used 0.005 with up to 10,000 trees on a GPU). Compare it with the clean script at
# the same learning rate.

# %%
import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from kaggle_utils import pull_competition_data, show_submissions  # symlink to lib/kaggle_utils.py

COMPETITION = "playground-series-s6e10"
data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb_allfeat.csv")

TARGET, ID, SEED = "satisfaction", "id", 42
train[TARGET] = train[TARGET].astype(int)

# %%
# Missing arrival delay: assume the flight arrived as late as it departed
for df in (train, test):
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])

CAT_FEATURES = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
]

# %%
# 1. Pair features: two columns joined into one string category (target-encoded inside each fold later)
PAIRS = {
    "Wifi_Booking_pair": ("Inflight wifi service", "Ease of Online booking"),
    "Wifi_Time_pair": ("Inflight wifi service", "Departure/Arrival time convenient"),
    "Boarding_Wifi_pair": ("Online boarding", "Inflight wifi service"),
    "Class_Travel_pair": ("Class", "Type of Travel"),
    "Cust_Travel_pair": ("Customer Type", "Type of Travel"),
    "Class_Cust_pair": ("Class", "Customer Type"),
    "Class_Wifi_pair": ("Class", "Inflight wifi service"),
    "Class_Boarding_pair": ("Class", "Online boarding"),
    "Travel_Wifi_pair": ("Type of Travel", "Inflight wifi service"),
    "Travel_Boarding_pair": ("Type of Travel", "Online boarding"),
    "Seat_Ent_pair": ("Seat comfort", "Inflight entertainment"),
    "Route_Class_pair": ("Flight Distance", "Class"),
    "Route_Travel_pair": ("Flight Distance", "Type of Travel"),
}
for df in (train, test):
    df["biz_elite_nj"] = ((df["Type of Travel"] == "Business travel") & (df["Class"] == "Business")).astype(np.int8)
    for name, (a, b) in PAIRS.items():
        df[name] = df[a].astype(str) + "_" + df[b].astype(str)
INTERACTION_COLS = list(PAIRS)

# 2. Digits of Age, Flight Distance and the delays (0 = ones digit, 1 = tens, ...)
DIGIT_MAP = {"Age": [0, 1], "Flight Distance": [0, 1, 2, 3],
             "Departure Delay in Minutes": [1], "Arrival Delay in Minutes": [1]}
digit_feats = []
for col, powers in DIGIT_MAP.items():
    for k in powers:
        name = f"{col}_digit{k}"
        for df in (train, test):
            df[name] = (df[col].fillna(0) // (10 ** k) % 10).astype(np.int8)
        digit_feats.append(name)

# 3. Route = Flight Distance. Row count per route (train + test) and mean ratings of the other rows on that route
for df in (train, test):
    df["Route_ID"] = df["Flight Distance"].astype(int)
route_counts = pd.concat([train["Route_ID"], test["Route_ID"]], ignore_index=True).value_counts()
for df in (train, test):
    df["Route_Freq"] = df["Route_ID"].map(route_counts).astype(np.int32)

profile_cols = ["Age", "Departure Delay in Minutes"] + RATING_COLS
route_profile = train.groupby("Route_ID")[profile_cols].mean()
route_profile.columns = [f"Route_Mean_{c.replace(' ', '_').replace('/', '_')}" for c in route_profile.columns]
route_profile = route_profile.reset_index()
train = train.merge(route_profile, on="Route_ID", how="left")
test = test.merge(route_profile, on="Route_ID", how="left")
for c in route_profile.columns.drop("Route_ID"):
    fill = train[c].mean()
    train[c] = train[c].fillna(fill)
    test[c] = test[c].fillna(fill)
train = train.drop(columns="Route_ID")
test = test.drop(columns="Route_ID")

# %%
# 4. Columns that get frequency-encoded (now) and target-encoded (inside each CV fold)
TARGET_ENCODE_COLS = CAT_FEATURES + RATING_COLS + digit_feats + INTERACTION_COLS
for col in ["Age", "Flight Distance"]:
    for df in (train, test):
        df[f"{col}_cat"] = df[col].fillna(-1).astype(str)
    TARGET_ENCODE_COLS.append(f"{col}_cat")

# everything target-encoded is treated as a string category
for df in (train, test):
    for col in CAT_FEATURES + digit_feats:
        df[col] = df[col].astype(str)
    for col in RATING_COLS:
        df[col] = "Rating_" + df[col].astype(str)

# Frequency encoding: share of ALL rows (train + test) that have this value
combined = pd.concat([train, test], ignore_index=True)
for col in TARGET_ENCODE_COLS:
    share = combined[col].value_counts(normalize=True).to_dict()
    train[f"{col}_FE"] = train[col].map(share).astype(np.float32)
    test[f"{col}_FE"] = test[col].map(share).astype(np.float32)
del combined
gc.collect()

# Pruning list from the public notebook
FEATURES_TO_DROP = [
    "Departure Delay in Minutes_digit1_FE", "Departure Delay in Minutes_digit1_TE_auto",
    "Arrival Delay in Minutes_digit1_FE", "Arrival Delay in Minutes_digit1_TE_auto",
    "Departure Delay in Minutes", "Flight Distance_digit3_FE", "Flight Distance_digit1_FE",
    "Age_digit0_FE", "Flight Distance_digit0_FE", "Flight Distance_digit2_FE",
]

FEATURES = [c for c in train.columns if c not in [ID, TARGET] + TARGET_ENCODE_COLS]
X = train[FEATURES + TARGET_ENCODE_COLS]
y = train[TARGET]
X_test = test[FEATURES + TARGET_ENCODE_COLS]
print(f"{len(FEATURES)} plain + {len(TARGET_ENCODE_COLS)} target-encoded columns")

# %%
# 5. 5-fold CV. The target encoder is fitted on each fold's training rows only (with its own internal
# 5-fold cross-fitting on those rows), then applied to the validation and test rows.
LEARNING_RATE = 0.05
xgb_params = {
    "n_estimators": 10000,
    "learning_rate": LEARNING_RATE,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.4,
    "colsample_bynode": 0.65,
    "min_child_weight": 30,
    "reg_lambda": 2.5,
    "max_bin": 256,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "random_state": SEED,
    "n_jobs": 4,
}


def encode_fold(X_tr, y_tr, others):
    """Target-encode TARGET_ENCODE_COLS using X_tr only; returns model-ready frames (raw string columns removed)."""
    te = TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED), smooth="auto")
    parts = [(X_tr, te.fit_transform(X_tr[TARGET_ENCODE_COLS], y_tr))]
    parts += [(o, te.transform(o[TARGET_ENCODE_COLS])) for o in others]
    out = []
    for frame, enc in parts:
        enc = pd.DataFrame(enc.astype(np.float32), columns=[f"{c}_TE_auto" for c in TARGET_ENCODE_COLS], index=frame.index)
        out.append(pd.concat([frame.drop(columns=TARGET_ENCODE_COLS), enc], axis=1).drop(columns=FEATURES_TO_DROP, errors="ignore"))
    return out


cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_scores, best_iters = [], []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    X_tr, X_va, X_te = encode_fold(X.iloc[tr_idx], y.iloc[tr_idx], [X.iloc[va_idx], X_test])
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_te)[:, 1] / cv.get_n_splits()
    fold_scores.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
    best_iters.append(model.best_iteration)
    print(f"Fold {fold}: ROC-AUC {fold_scores[-1]:.5f}  best_iteration {model.best_iteration}", flush=True)
    del X_tr, X_va, X_te
    gc.collect()

oof_auc = roc_auc_score(y, oof)
n_final_features = model.n_features_in_
print(f"\nOOF ROC-AUC: {oof_auc:.5f}")
print(f"Fold ROC-AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}  ({n_final_features} features)")

np.save(os.path.join(data_dir, "oof_xgb_allfeat.npy"), oof)
np.save(os.path.join(data_dir, "test_xgb_allfeat.npy"), test_pred)

# %%
# Feature importance (gain), last fold model
importance = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)
print(importance.head(25).to_string())

# %%
submission = pd.DataFrame({"id": test[ID], "satisfaction": test_pred})  # probabilities, ROC-AUC is the metric
submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")

# %%
# Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition=COMPETITION,
    message="XGBoost, all public-notebook features (frequency, route, digit, target-encoded pairs), lr 0.05",
    wait_for_score=True,
    timeout=60,
)

# %%
show_submissions(competition=COMPETITION, n=10)

# %%
# Log this run (read it back with load_runs())
from kaggle_utils import log_run

log_run(
    COMPETITION, "xgb-allfeat", metric="roc_auc",
    cv_mean=np.mean(fold_scores), cv_std=np.std(fold_scores),
    params=xgb_params, seed=SEED, features=n_final_features,
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    notes="Includes frequency/route/digit features that break the logical-features rule; experiment only.",
    extra={"oof_auc": oof_auc, "cv": "5-fold stratified", "best_iterations": best_iters},
)
