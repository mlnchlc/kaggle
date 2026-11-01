# %% [markdown]
# # Airline satisfaction: "Single XGBoost Model" (public notebook, ported)
#
# A port of the public Kaggle notebook "Single XGBoost Model" for Playground S6E10 (public CV 0.96096 AUC),
# kept here as a reference to compare against `playground-series-s6e10-airline-xgb` (the clean model) and
# `playground-series-s6e10-airline-xgb-allfeat` (the same feature recipe with a faster learning rate).
#
# It uses features that break this repo's "logical features only" rule (frequency encoding over train + test,
# route IDs from Flight Distance, digit features). It is a reference, not the model to ship.
#
# Credit, as given by the notebook's author: Chris Deotte (EDA and XGB starter notebooks), Dmitriy Kolmikov
# (target encoding of flight distances), Sarun P M (route-based profiling), Shelton Wang and Evgen Dvorkin
# (persona-context interactions), Starkhushi (statistical field guide, noisy columns).
#
# Changes from the original: data comes from `../data/<slug>/`, the device is set below (the original ran on a
# GPU: ~47 min; on this 4-core CPU the 0.005 learning rate takes many hours), and the run is logged with `log_run`.

# %%
import gc
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from kaggle_utils import pull_competition_data, show_submissions  # symlink to lib/kaggle_utils.py

DEVICE = "cpu"          # set to "cuda" on a machine with a GPU
LEARNING_RATE = 0.005   # the original value; use 0.05 for a quick CPU run

COMPETITION = "playground-series-s6e10"
data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb_top.csv")

TARGET, ID, EXPERIMENT_NAME, NJ_SEED = "satisfaction", "id", "XGBoost", 42
train[TARGET] = train[TARGET].astype(int)

for df in (train, test):
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])

CAT_FEATURES = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
]

# %%
# Feature engineering: domain flag and pair features (two columns joined into one string category)
for df in (train, test):
    df["biz_elite_nj"] = ((df["Type of Travel"] == "Business travel") & (df["Class"] == "Business")).astype(np.int8)
    df["Wifi_Booking_pair"] = df["Inflight wifi service"].astype(str) + "_" + df["Ease of Online booking"].astype(str)
    df["Wifi_Time_pair"] = df["Inflight wifi service"].astype(str) + "_" + df["Departure/Arrival time convenient"].astype(str)
    df["Boarding_Wifi_pair"] = df["Online boarding"].astype(str) + "_" + df["Inflight wifi service"].astype(str)
    df["Class_Travel_pair"] = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)
    df["Cust_Travel_pair"] = df["Customer Type"].astype(str) + "_" + df["Type of Travel"].astype(str)
    df["Class_Cust_pair"] = df["Class"].astype(str) + "_" + df["Customer Type"].astype(str)
    df["Class_Wifi_pair"] = df["Class"].astype(str) + "_" + df["Inflight wifi service"].astype(str)
    df["Class_Boarding_pair"] = df["Class"].astype(str) + "_" + df["Online boarding"].astype(str)
    df["Travel_Wifi_pair"] = df["Type of Travel"].astype(str) + "_" + df["Inflight wifi service"].astype(str)
    df["Travel_Boarding_pair"] = df["Type of Travel"].astype(str) + "_" + df["Online boarding"].astype(str)
    df["Seat_Ent_pair"] = df["Seat comfort"].astype(str) + "_" + df["Inflight entertainment"].astype(str)
    df["Route_Class_pair"] = df["Flight Distance"].astype(str) + "_" + df["Class"].astype(str)
    df["Route_Travel_pair"] = df["Flight Distance"].astype(str) + "_" + df["Type of Travel"].astype(str)

INTERACTION_COLS = [
    "Wifi_Booking_pair", "Wifi_Time_pair", "Boarding_Wifi_pair", "Class_Travel_pair", "Cust_Travel_pair",
    "Class_Cust_pair", "Class_Wifi_pair", "Class_Boarding_pair", "Travel_Wifi_pair", "Travel_Boarding_pair",
    "Seat_Ent_pair", "Route_Class_pair", "Route_Travel_pair",
]

# Digits of Age, Flight Distance and the delays
DIGIT_MAP = {"Age": [0, 1], "Flight Distance": [0, 1, 2, 3], "Departure Delay in Minutes": [1], "Arrival Delay in Minutes": [1]}
digit_feats_nj = []
for c, powers in DIGIT_MAP.items():
    for k in powers:
        col_name = f"{c}_digit{k}"
        train[col_name] = (train[c].fillna(0) // (10 ** k) % 10).astype(np.int8)
        test[col_name] = (test[c].fillna(0) // (10 ** k) % 10).astype(np.int8)
        digit_feats_nj.append(col_name)

# Route = Flight Distance: row count per route over train + test, and mean ratings per route
for df in (train, test):
    df["Route_ID"] = df["Flight Distance"].astype(int)

route_counts = pd.concat([train["Route_ID"], test["Route_ID"]], ignore_index=True).value_counts()
for df in (train, test):
    df["Route_Freq"] = df["Route_ID"].map(route_counts).astype(np.int32)

route_profile_cols = ["Age", "Departure Delay in Minutes"] + RATING_COLS
route_profile = train.groupby("Route_ID")[route_profile_cols].mean()
route_profile.columns = [f"Route_Mean_{c.replace(' ', '_').replace('/', '_')}" for c in route_profile.columns]
route_profile = route_profile.reset_index()
train = train.merge(route_profile, on="Route_ID", how="left")
test = test.merge(route_profile, on="Route_ID", how="left")
for c in route_profile.columns:
    if c == "Route_ID":
        continue
    fill = train[c].mean()
    train[c] = train[c].fillna(fill)
    test[c] = test[c].fillna(fill)
train = train.drop(columns=["Route_ID"])
test = test.drop(columns=["Route_ID"])

# Columns that get frequency-encoded now and target-encoded inside each fold
TARGET_ENCODE_COLS = CAT_FEATURES + RATING_COLS + digit_feats_nj + INTERACTION_COLS
for c in ["Age", "Flight Distance"]:
    cat_name = f"{c}_cat"
    train[cat_name] = train[c].fillna(-1).astype(str)
    test[cat_name] = test[c].fillna(-1).astype(str)
    TARGET_ENCODE_COLS.append(cat_name)

for df in (train, test):
    for col in CAT_FEATURES + digit_feats_nj:
        df[col] = df[col].astype(str)
    for col in RATING_COLS:
        df[col] = "Rating_" + df[col].astype(str)

# Frequency encoding: share of ALL rows (train + test) with this value
combined_nj = pd.concat([train, test], ignore_index=True)
for col in TARGET_ENCODE_COLS:
    freq_mapping = combined_nj[col].value_counts(normalize=True).to_dict()
    train[f"{col}_FE"] = train[col].map(freq_mapping).astype(np.float32)
    test[f"{col}_FE"] = test[col].map(freq_mapping).astype(np.float32)
del combined_nj
gc.collect()

# Pruning list from the original notebook
FEATURES_TO_DROP = [
    "Departure Delay in Minutes_digit1_FE", "Departure Delay in Minutes_digit1_TE_auto",
    "Arrival Delay in Minutes_digit1_FE", "Arrival Delay in Minutes_digit1_TE_auto",
    "Departure Delay in Minutes", "Flight Distance_digit3_FE", "Flight Distance_digit1_FE",
    "Age_digit0_FE", "Flight Distance_digit0_FE", "Flight Distance_digit2_FE",
]

# %%
# 5-fold training. The target encoder is fitted on each fold's training rows only.
FEATURES = [c for c in train.columns if c not in [ID, TARGET] + TARGET_ENCODE_COLS]
X = train[FEATURES + TARGET_ENCODE_COLS]
y = train[TARGET]
X_test = test[FEATURES + TARGET_ENCODE_COLS]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=NJ_SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores, best_iters = [], []

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
    "device": DEVICE,
    "eval_metric": "auc",
    "early_stopping_rounds": 200,
    "random_state": NJ_SEED,
}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx].copy(), y.iloc[valid_idx]
    X_test_fold = X_test.copy()

    te_auto = TargetEncoder(shuffle=True, cv=5, smooth="auto", random_state=NJ_SEED)
    X_train_enc = te_auto.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_valid_enc = te_auto.transform(X_valid[TARGET_ENCODE_COLS])
    X_test_enc = te_auto.transform(X_test_fold[TARGET_ENCODE_COLS])

    for i, col in enumerate(TARGET_ENCODE_COLS):
        X_train[f"{col}_TE_auto"] = X_train_enc[:, i].astype(np.float32)
        X_valid[f"{col}_TE_auto"] = X_valid_enc[:, i].astype(np.float32)
        X_test_fold[f"{col}_TE_auto"] = X_test_enc[:, i].astype(np.float32)

    del X_train_enc, X_valid_enc, X_test_enc
    gc.collect()

    # remove the raw string columns (now encoded) and the pruned features
    drop = TARGET_ENCODE_COLS + FEATURES_TO_DROP
    X_train = X_train.drop(columns=drop, errors="ignore")
    X_valid = X_valid.drop(columns=drop, errors="ignore")
    X_test_fold = X_test_fold.drop(columns=drop, errors="ignore")

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=1000)

    valid_probs = model.predict_proba(X_valid)[:, 1]
    oof_preds[valid_idx] = valid_probs
    test_preds += model.predict_proba(X_test_fold)[:, 1] / skf.n_splits

    fold_scores.append(roc_auc_score(y_valid, valid_probs))
    best_iters.append(model.best_iteration)
    print(f"Fold {fold} | Best Iter: {model.best_iteration} | ROC-AUC: {fold_scores[-1]:.5f}", flush=True)

mean_auc = np.mean(fold_scores)
print(f"\n{EXPERIMENT_NAME} CV ROC-AUC: {mean_auc:.5f} +/- {np.std(fold_scores):.5f}")

np.save(os.path.join(data_dir, "oof_xgb_top.npy"), oof_preds)
np.save(os.path.join(data_dir, "test_xgb_top.npy"), test_preds)

# %%
submission = pd.DataFrame({ID: test[ID], TARGET: test_preds})  # probabilities, ROC-AUC is the metric
submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")

# %%
# Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition=COMPETITION,
    message=f"Public 'Single XGBoost Model' port, lr {LEARNING_RATE}, {DEVICE}",
    wait_for_score=True,
    timeout=60,
)

# %%
show_submissions(competition=COMPETITION, n=10)

# %%
# Log this run (read it back with load_runs())
from kaggle_utils import log_run

log_run(
    COMPETITION, "xgb-top", metric="roc_auc",
    cv_mean=mean_auc, cv_std=np.std(fold_scores),
    params=xgb_params, seed=NJ_SEED, features=model.n_features_in_,
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    notes="Port of a public notebook; includes frequency/route/digit features that break the logical-features rule.",
    extra={"oof_auc": roc_auc_score(y, oof_preds), "cv": "5-fold stratified", "best_iterations": best_iters},
)
