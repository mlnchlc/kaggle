# %%
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

# %%
import os
from kaggle_utils import pull_competition_data  # kaggle_utils.py is a symlink to the repo-root copy

# Paths are relative to this notebook's folder
data_dir = os.path.join("..", "data", "titanic")
pull_competition_data("titanic", data_dir=data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
kaggle_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
kaggle_output_file = os.path.join(data_dir, "submission_xgb.csv")
train_df.info()
kaggle_test_df.info()

print(train_df.head(5))

# %%

# 1. Separate features and target
X = train_df.drop(columns=["Survived", "PassengerId"])
X_kaggle_test = kaggle_test_df.drop(columns=["PassengerId"])

# Keep Name for feature engineering.
# Ticket and Cabin can remain too if you want;
# the ColumnTransformer will select only listed features.

y = train_df['Survived']

# 1. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# %%
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, target_col):
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group_cols)[self.target_col].median()
        self.median_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        group_medians = X.set_index(self.group_cols).index.map(self.medians_)

        X[self.target_col] = (
            X[self.target_col]
            .fillna(pd.Series(group_medians, index=X.index))
            .fillna(self.median_)
        )
        return X

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Extract passenger title
        X['Title'] = (
            X['Name']
            .str.extract(r',\s*([^.]*)\.', expand=False)
            .str.strip()
        )


        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Officer',
            'Rev': 'Officer',
            'Col': 'Officer',
            'Major': 'Officer',
            'Capt': 'Officer',
            'Mlle': 'Miss',
            'Mme': 'Mrs',
            'Ms': 'Miss',
            'Lady': 'Royalty',
            'Sir': 'Royalty',
            'Countess': 'Royalty',
            'Jonkheer': 'Royalty',
            'Don': 'Royalty',
            'Dona': 'Royalty'
        }

        X['Title'] = X['Title'].map(title_mapping).fillna('Other')
        # Family features
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        # X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        # IsChild is added by AddIsChild, after Age has been imputed
        return X


class AddIsChild(BaseEstimator, TransformerMixin):
    """Flags children. Must run AFTER age imputation: NaN <= 12 is False,
    so computing it earlier silently marks every passenger with a missing Age as an adult."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['IsChild'] = (X['Age'] <= 12).astype(int)
        return X


numeric_features = [
    "Age", "SibSp", "Parch", "Fare",
    "FamilySize" , #"IsAlone"
]

categorical_features = ["Sex", "Embarked", "Title", "Pclass", "IsChild"] 

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
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
        ('features', TitanicFeatureEngineer()),
        ('age_imputer', GroupMedianImputer(
            group_cols=['Title', 'Pclass'],
            target_col='Age'
        )),
        ('is_child', AddIsChild()),
        ('fare_imputer', GroupMedianImputer(
            group_cols=['Pclass'],
            target_col='Fare'
        )),
        # clone() gives each pipeline its own unfitted copy; sharing one preprocessor object
        # means fitting one pipeline silently re-fits the preprocessor inside another
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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        build_pipeline(params),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    return scores.mean()


# %%
# 4. Run the search
from datetime import datetime

# Every trial is stored in <data_dir>/optuna.db (gitignored). One study per run, named by timestamp, so trials
# from different feature sets or CV setups never mix. Reload later: optuna.load_study(study_name=..., storage=...)
STUDY_NAME = f"titanic-xgb-{datetime.now():%Y%m%d-%H%M%S}"
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage="sqlite:///" + os.path.join(data_dir, "optuna.db"),
    direction="maximize",
)
study.optimize(objective, n_trials=30)

print("Best CV accuracy:", study.best_value)
print("Best parameters:", study.best_params)

# %%
# 5. Train the best pipeline on ALL training data for submission
best_model = build_pipeline(study.best_params)
best_model.fit(X, y)

# %%
# Same metric Optuna optimised; this also shows the spread across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    build_pipeline(study.best_params),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("Fold accuracies:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("Std CV accuracy:", cv_scores.std())

# %%
# 6. Evaluate once on the untouched holdout set (using a model fit on X_train only)
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
predictions = best_model.predict(X_kaggle_test)
probabilities = best_model.predict_proba(X_kaggle_test)[:, 1]

submission = pd.DataFrame({'PassengerId': kaggle_test_df['PassengerId'], 'Survived': predictions})
submission.to_csv(kaggle_output_file, index=False)

# %%
# Log this run (score history lives in runs.jsonl at the repo root; read it with load_runs()).
# This script does not submit to Kaggle, so there is no public score; add one by hand if you submit.
from kaggle_utils import log_run

log_run(
    "titanic", "xgb", metric="accuracy",
    cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
    params=study.best_params, seed=42, features=X.shape[1],
    submission_file=kaggle_output_file,
    extra={"cv": "5-fold stratified on the training split", "tune_best_accuracy": study.best_value,
           "tune_trials": len(study.trials), "optuna_study": STUDY_NAME},
)
