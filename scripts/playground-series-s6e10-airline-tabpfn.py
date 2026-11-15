# %%
import os
# Work from this file's folder even when an editor (e.g. Zed's REPL) starts the kernel in the repo root,
# so the kaggle_utils symlink and the ../data paths resolve.
if os.path.isfile(os.path.join("scripts", "kaggle_utils.py")):
    os.chdir("scripts")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

# TabPFN (Prior-Data Fitted Network for Tabular Classification)
from tabpfn import TabPFNClassifier

# TabPFN v2 requires a free one-time API key to download the pre-trained weights:
# 1. Sign in at https://ux.priorlabs.ai/
# 2. Copy your key from https://ux.priorlabs.ai/account
# 3. Put it in the repo-root .env file (git-ignored) as TABPFN_TOKEN=... / TABPFN_API_KEY=...
# Never paste keys directly into a notebook: it's easy to commit them by accident.
from dotenv import load_dotenv
load_dotenv()  # searches upward from the working directory for .env
assert os.getenv("TABPFN_TOKEN"), "TABPFN_TOKEN not set - add it to .env"

# %%
from kaggle_utils import pull_competition_data  # kaggle_utils.py is a symlink to the repo-root copy

# Paths are relative to this notebook's folder
data_dir = os.path.join("..", "data", "playground-series-s6e10")
pull_competition_data("playground-series-s6e10", data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
submission_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_tabpfn.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {kaggle_test_df.shape}")

 # %%
# 1. Separate features and target
X = train_df.drop(columns=["satisfaction", "id"])
X_kaggle_test = kaggle_test_df.drop(columns=["id"], errors="ignore")
y = train_df['satisfaction'].astype(int)

# Stratified split before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# Note on TabPFN sample size:
# TabPFN is a transformer foundation model that processes context in-memory.
# For fast iteration and baseline verification on large tabular datasets (700k rows),
# training on a representative stratified subsample (e.g., 5,000 - 20,000 rows)
# or using an ensemble of subsamples is standard and provides strong generalization.
SAMPLE_SIZE = 10000  # Set to None to use full data or adjust for quick experiments
if SAMPLE_SIZE and len(X_train) > SAMPLE_SIZE:
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=SAMPLE_SIZE, random_state=42, stratify=y_train
    )
    print(f"Using stratified subsample for TabPFN training: {X_train_sub.shape}")
else:
    X_train_sub, y_train_sub = X_train, y_train


# %%
class AirlineFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # =======================================================
        # Baseline: Feature engineering commented out
        # =======================================================
        # 1. Delay features
        # X['TotalDelay'] = X['Departure Delay in Minutes'] + X['Arrival Delay in Minutes']
        # X['HasDepartureDelay'] = (X['Departure Delay in Minutes'] > 0).astype(int)
        # X['HasArrivalDelay'] = (X['Arrival Delay in Minutes'] > 0).astype(int)
        # X['DelayRatio'] = X['Arrival Delay in Minutes'] / (X['Departure Delay in Minutes'] + 1)

        # 2. Survey score aggregations
        # survey_cols = [
        #     'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        #     'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        #     'Inflight entertainment', 'On-board service', 'Leg room service',
        #     'Baggage handling', 'Checkin service', 'Cleanliness'
        # ]
        # X['SurveyTotal'] = X[survey_cols].sum(axis=1)
        # X['SurveyMean'] = X[survey_cols].mean(axis=1)
        # X['SurveyStd'] = X[survey_cols].std(axis=1)

        # Drop identifier if present
        X = X.drop(columns=['id'], errors='ignore')

        return X


# Variable lists for Airline Passenger Satisfaction
numeric_features = [
    "Age", "Flight Distance", "Inflight wifi service",
    "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service",
    "Leg room service", "Baggage handling", "Checkin service",
    "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes"
]

categorical_features = [
    "Gender", "Customer Type", "Type of Travel", "Class"
]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# %%
# Define TabPFN Pipeline
# TabPFN is a zero-shot foundation model: it requires no hyperparameter tuning
tabpfn_pipeline = Pipeline([
    ('features', AirlineFeatureEngineer()),
    ('preprocessing', clone(preprocessor)),
    ('classifier', TabPFNClassifier(
        device='auto',           # uses CUDA GPU if available, else CPU
        ignore_pretraining_limits=True, # Allow >5000 samples on CPU
        n_estimators=4,          # number of forward passes in ensemble (default: auto or 4-8)
        random_state=42
    ))
])

# %%
# 3. Evaluate Baseline TabPFN with 5-Fold Cross-Validation
# Using the subsample for fast validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    tabpfn_pipeline,
    X_train_sub,
    y_train_sub,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1
)

print("TabPFN 5-Fold CV ROC-AUC Scores:", np.round(cv_scores, 5))
print(f"Mean CV ROC-AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")

# %%
# 4. Fit TabPFN model and evaluate on holdout set
tabpfn_pipeline.fit(X_train_sub, y_train_sub)

# Evaluate on holdout (predict in chunks to manage memory)
def predict_proba_batched(model, X_data, batch_size=5000):
    probs = []
    for start in range(0, len(X_data), batch_size):
        end = min(start + batch_size, len(X_data))
        batch = X_data.iloc[start:end]
        p = model.predict_proba(batch)[:, 1]
        probs.append(p)
    return np.concatenate(probs)

holdout_probs = predict_proba_batched(tabpfn_pipeline, X_test, batch_size=5000)
holdout_preds = (holdout_probs > 0.5).astype(int)

print(f"Holdout Test ROC-AUC: {roc_auc_score(y_test, holdout_probs):.5f}")
print()
print("Classification Report:")
print(classification_report(y_test, holdout_preds))

# %%
# 5. Generate Predictions for Kaggle Test Set
print(f"Generating predictions for {len(X_kaggle_test)} test rows...")
kaggle_test_probs = predict_proba_batched(tabpfn_pipeline, X_kaggle_test, batch_size=5000)

submission = pd.DataFrame({
    'id': kaggle_test_df['id'],
    'satisfaction': kaggle_test_probs   # Predicted probability of satisfaction
})

submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")
print("Preview:")
print(submission.head())

# %%
# 6. Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition="playground-series-s6e10",
    message="TabPFN baseline model with Stratified CV",
    wait_for_score=True,
    timeout=60
)

# %%
# 7. Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs())
from kaggle_utils import log_run

log_run(
    "playground-series-s6e10", "tabpfn", metric="roc_auc",
    cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
    params={"n_estimators": 4, "sample_size": SAMPLE_SIZE}, seed=42, features=X.shape[1],
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    extra={"holdout_roc_auc": roc_auc_score(y_test, holdout_probs), "train_rows": len(X_train_sub),
           "cv": "5-fold stratified on the training subsample"},
)
