# %%
import optuna
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
import shap
import numpy as np

# %%
import os
from kaggle_utils import pull_competition_data  # kaggle_utils.py is a symlink to the repo-root copy

# Auto-pull competition data (skips if train.csv and test.csv already exist)
# Paths are relative to this notebook's folder (VS Code/Jupyter run notebooks from their own directory)
data_dir = os.path.join('..', 'data', 'spaceship-titanic')
pull_competition_data('spaceship-titanic', data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
submission_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
kaggle_output_file = os.path.join(data_dir, 'submission_xgb.csv')
train_df.info()

# %%
# 1. Separate features and target
X = train_df.drop(columns=["Transported"])
X_kaggle_test = kaggle_test_df.copy()

y = train_df['Transported'].astype(int)

# Split before fitting any preprocessing.
# Passengers in the same group (PassengerId prefix) travel together and share outcomes,
# and GroupSize is computed from the rows present. A plain train_test_split would put
# family members on both sides (~41% of holdout rows), leaking signal and corrupting GroupSize.
# Taking one fold of StratifiedGroupKFold gives an 80/20 split that keeps groups whole
# and preserves class balance - matching how the Kaggle test set is built (no shared groups).
groups_all = X['PassengerId'].str.split('_').str[0]
holdout_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(holdout_split.split(X, y, groups=groups_all))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train: {len(X_train)} rows, holdout: {len(X_test)} rows")


# %%
class SpaceshipTitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Group features from PassengerId
        X['GroupId'] = X['PassengerId'].str.split('_').str[0]
        X['GroupSize'] = X.groupby('GroupId')['GroupId'].transform('count')

        # 2. Split Cabin into Deck, Num, Side, and physical CabinRegion
        cabin_extracted = X['Cabin'].str.extract(r'^(?P<Deck>[A-Za-z])?/(?P<Num>\d+)?/(?P<Side>[A-Za-z])?')
        X[['Deck', 'Num', 'Side']] = cabin_extracted[['Deck', 'Num', 'Side']]
        X['Num'] = pd.to_numeric(X['Num'])
        X['CabinRegion'] = X['Num'] // 300

        # 3. Spending features - Impute missing spending as 0
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X[spend_cols] = X[spend_cols].fillna(0)

        X['TotalSpending'] = X[spend_cols].sum(axis=1)
        X['LuxurySpending'] = X[['Spa', 'VRDeck', 'RoomService']].sum(axis=1)
        X['CommonSpending'] = X[['FoodCourt', 'ShoppingMall']].sum(axis=1)
        X['LuxuryToCommonRatio'] = X['LuxurySpending'] / (X['CommonSpending'] + 1)
        X['NoSpending'] = (X['TotalSpending'] == 0).astype(int)
        
        # 4. Log-transform spending columns
        for col in spend_cols:
            X[f'{col}_log'] = np.log1p(X[col])

        # Drop columns not needed for modeling
        X = X.drop(columns=['PassengerId', 'GroupId', 'Cabin', 'Name'], errors='ignore')

        return X


# Numeric features with new engineered features added
numeric_features = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall",
    "Spa", "VRDeck", "Num", "CabinRegion", "TotalSpending",
    "RoomService_log", "FoodCourt_log", "ShoppingMall_log",
    "Spa_log", "VRDeck_log", 'LuxurySpending', 'CommonSpending',
    'LuxuryToCommonRatio', 'GroupSize', 'NoSpending'
]

categorical_features = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

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
# 3. Define the hyperparameter search
def build_pipeline(params):
    """One place that defines the model, so tuning, evaluation and submission all use the same pipeline."""
    return Pipeline([
        ('features', SpaceshipTitanicFeatureEngineer()),
        ('preprocessing', clone(preprocessor)),
        ('classifier', XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        )),
    ])


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1, 10
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10, log=True
        ),
    }

    # Group by family ID so groups are never split across folds
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = X_train['PassengerId'].str.split('_').str[0]

    scores = cross_val_score(
        build_pipeline(params),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        groups=groups,
        n_jobs=1
    )

    return scores.mean()


# %%

# 4. Run the search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best CV accuracy:", study.best_value)
print("Best parameters:", study.best_params)

# %%
# 5. Train the best pipeline on ALL training data for submission
best_model = build_pipeline(study.best_params)
best_model.fit(X, y)

# %%

# Cross-validation scores (from Optuna search above)
# The CV was already done inside the objective function.
# best_model is now fit on ALL data for final Kaggle submission.
# See eval_model cell below for holdout evaluation.
print("Best CV accuracy from Optuna:", study.best_value)


# %%
# 6. Evaluate on holdout set (using a model fit on X_train only)
# Note: best_model above is fit on ALL data for submission.
# For a fair holdout evaluation, we fit a separate copy on X_train:
eval_model = build_pipeline(study.best_params)
eval_model.fit(X_train, y_train)

predictions = eval_model.predict(X_test)
probabilities = eval_model.predict_proba(X_test)[:, 1]

print("Test ROC-AUC:", roc_auc_score(y_test, probabilities))
print(classification_report(y_test, predictions))

# %%
# Get the trained XGBoost classifier
xgb_model = best_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = best_model.named_steps[
    "preprocessing"
].get_feature_names_out()

# Pair names with importance scores
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.feature_importances_
})

# Sort from most to least important
importance = importance.sort_values(
    "Importance", ascending=False
)

print(importance.to_string(index=False))

# %%
# Explain the holdout set with eval_model (best_model was trained on these rows too)
X_transformed = eval_model[:-1].transform(X_test)  # all pipeline steps except classifier
feature_names = eval_model.named_steps["preprocessing"].get_feature_names_out()

# Compute SHAP values
explainer = shap.TreeExplainer(eval_model.named_steps["classifier"])
shap_values = explainer.shap_values(X_transformed)

# Summary plot — shows direction + magnitude for every feature
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)

# %%
predictions = best_model.predict(X_kaggle_test)
probabilities = best_model.predict_proba(X_kaggle_test)[:, 1]

submission = pd.DataFrame({
    'PassengerId': kaggle_test_df['PassengerId'], 
    'Transported': predictions.astype(bool)   # <-- convert 0/1 back to False/True
})

submission.to_csv(kaggle_output_file, index=False)
print(f"Saved submission to {kaggle_output_file} ({len(submission)} rows)")


# %%
# 7. Auto-submit to Kaggle and retrieve public score
from kaggle_utils import submit_prediction

submission_result = submit_prediction(
    file_path=kaggle_output_file,
    competition="spaceship-titanic",
    message="XGBoost pipeline with Optuna tuning and feature engineering",
    wait_for_score=True,
    timeout=60
)

# %%
# 8. Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs())
from kaggle_utils import log_run

log_run(
    "spaceship-titanic", "xgb", metric="accuracy",
    cv_mean=study.best_value, params=study.best_params, seed=42, features=X.shape[1],
    public_score=getattr(submission_result, "public_score", None),
    submission_file=kaggle_output_file,
    extra={"cv": "5-fold StratifiedGroupKFold (best Optuna trial)", "tune_trials": len(study.trials)},
)
