
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



# 1. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# 3. Define the hyperparameter search
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

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        )),
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    return scores.mean()


# 4. Run the search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best CV ROC-AUC:", study.best_value)
print("Best parameters:", study.best_params)

# 5. Train the best pipeline on all training data
best_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", XGBClassifier(
        **study.best_params,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )),
])

best_model.fit(X_train, y_train)

# 6. Evaluate once on the untouched test set
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

print("Test ROC-AUC:", roc_auc_score(y_test, probabilities))
print(classification_report(y_test, predictions))
