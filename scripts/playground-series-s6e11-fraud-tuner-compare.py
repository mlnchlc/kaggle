# %% [markdown]
# # Playground S6E11: does Optuna's TPE beat random search? (tuner comparison)
#
# An experiment, not a submission: nothing is sent to Kaggle and nothing is written to `runs.jsonl`.
#
# Both samplers get the SAME search space, tuning subsample (100k rows, 3-fold, learning rate 0.1) and trial budget
# (30 trials), over 5 sampler seeds each (0-4). The data subsample and the CV folds are fixed, so only the sampler's
# randomness changes between seeds.
#   * TPESampler uses a MedianPruner (as in the other S6E11 scripts), so some of its trials stop early.
#   * RandomSampler runs with NO pruning: every trial is a full 3-fold run. This is a real difference in method and in
#     cost, not just in sampling, so the comparison is "TPE + pruning" vs "plain random search".
# The tuning score is noisy (it is a 3-fold score on a 100k subsample), so each seed's best parameters are re-scored with
# one common 5-fold run on the FULL training data at learning rate 0.05. That second number is the one to trust.
# Raw features only. Set SMOKE=1 for a tiny check run.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from kaggle_utils import pull_competition_data  # symlink to lib/kaggle_utils.py

COMPETITION = "playground-series-s6e11"
SEED = 42                      # fixes the data subsample and the CV folds; the SAMPLER seeds are SAMPLER_SEEDS
SMOKE = bool(os.environ.get("SMOKE"))
SAMPLER_SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
N_TRIALS = 3 if SMOKE else 30
TUNE_SAMPLE_SIZE = 5_000 if SMOKE else 100_000
TUNE_LEARNING_RATE = 0.1
FINAL_LEARNING_RATE = 0.05
MAX_TREES = 40 if SMOKE else 5000

optuna.logging.set_verbosity(optuna.logging.WARNING)

data_dir = os.path.join("..", "data", COMPETITION)
pull_competition_data(COMPETITION, data_dir=data_dir)
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
ID_COL = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"), nrows=1).columns[0]
TARGET = next(c for c in train_df.columns if c not in test_df.columns)
if SMOKE:
    train_df = train_df.sample(20_000, random_state=SEED).reset_index(drop=True)

X = train_df.drop(columns=[TARGET, ID_COL])
y = train_df[TARGET].astype(int)
for col in X.select_dtypes(include=["object", "string", "category"]).columns:
    X[col] = pd.Categorical(X[col], categories=X[col].astype("category").cat.categories)
print(f"train={X.shape} target={TARGET!r} sampler seeds={SAMPLER_SEEDS} trials={N_TRIALS}")

# %%
def run_cv(params, X, y, n_splits, trial=None):
    """Stratified K-fold with early stopping; returns the OOF ROC-AUC (same function as the other S6E11 scripts)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    fold_scores = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        model = XGBClassifier(**params, n_estimators=MAX_TREES, early_stopping_rounds=100, eval_metric="auc",
                              tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=4)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(roc_auc_score(y.iloc[va], oof[va]))
        if trial is not None:  # pruning hook (TPE runs only)
            trial.report(np.mean(fold_scores), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return roc_auc_score(y, oof), fold_scores


X_tune, _, y_tune, _ = train_test_split(X, y, train_size=min(TUNE_SAMPLE_SIZE, len(X) // 2),
                                        random_state=SEED, stratify=y)


def suggest(trial):
    """The search space: identical to the baseline script (max_depth 2-10)."""
    return {
        "learning_rate": TUNE_LEARNING_RATE,
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def make_study(kind, seed, stamp):
    storage = "sqlite:///" + os.path.join(data_dir, "optuna.db")
    name = f"{COMPETITION}-tuner-{kind}-seed{seed}-{stamp}"
    if kind == "tpe":
        return optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=seed),
                                   pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    return optuna.create_study(study_name=name, storage=storage, direction="maximize",
                               sampler=optuna.samplers.RandomSampler(seed=seed),
                               pruner=optuna.pruners.NopPruner())


# %%
# Tune: 5 seeds x 2 samplers x 30 trials, one stored study each (named by sampler, seed and timestamp).
STAMP = f"{datetime.now():%Y%m%d-%H%M%S}"
studies = {"tpe": {}, "random": {}}
for seed in SAMPLER_SEEDS:
    for kind in ("tpe", "random"):
        study = make_study(kind, seed, STAMP)
        # Random search gets no pruning (trial=None), TPE gets the MedianPruner through trial.report
        use_trial = kind == "tpe"
        study.optimize(lambda t: run_cv(suggest(t), X_tune, y_tune, 3, trial=t if use_trial else None)[0],
                       n_trials=N_TRIALS)
        studies[kind][seed] = study
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"{kind:6s} seed {seed}: best tuning score {study.best_value:.5f} (pruned {n_pruned}/{N_TRIALS})", flush=True)

# %%
# Best-so-far tuning score by trial number (only COMPLETE trials have a final score; pruned ones carry the last best).
def best_so_far(study):
    best, curve = -np.inf, []
    for t in sorted(study.trials, key=lambda t: t.number):
        if t.state == optuna.trial.TrialState.COMPLETE:
            best = max(best, t.value)
        curve.append(best)
    return curve


curves = {k: np.array([best_so_far(s) for s in studies[k].values()]) for k in studies}
checkpoints = [n for n in (1, 5, 10, 15, 20, 25, 30) if n <= N_TRIALS]
curve_table = pd.DataFrame({k: [curves[k][:, n - 1].mean() for n in checkpoints] for k in curves},
                           index=pd.Index(checkpoints, name="trial"))
print("mean best-so-far tuning score over seeds:\n", curve_table.round(5).to_string())

os.makedirs("output", exist_ok=True)
plt.figure(figsize=(6, 4))
for kind, arr in curves.items():
    plt.plot(range(1, N_TRIALS + 1), arr.mean(axis=0), label=kind)
    plt.fill_between(range(1, N_TRIALS + 1), arr.min(axis=0), arr.max(axis=0), alpha=0.15)
plt.xlabel("trial")
plt.ylabel("best tuning ROC-AUC so far")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("output", "tuner_compare_curves.png"), dpi=120)
plt.show()

# %%
# Re-score every seed's best parameters with ONE common 5-fold run on the full data at the lower learning rate.
rows = []
for kind in ("tpe", "random"):
    for seed in SAMPLER_SEEDS:
        study = studies[kind][seed]
        params = {**study.best_params, "learning_rate": FINAL_LEARNING_RATE}
        full_score, _ = run_cv(params, X, y, 5)
        rows.append({"sampler": kind, "seed": seed, "tuning_score": study.best_value, "full_data_cv": full_score,
                     "max_depth": study.best_params["max_depth"]})
        print(f"{kind:6s} seed {seed}: tuning {study.best_value:.5f}  full-data 5-fold {full_score:.5f}", flush=True)
results = pd.DataFrame(rows)
print(results.round(5).to_string(index=False))

# %%
summary = results.groupby("sampler")[["tuning_score", "full_data_cv"]].agg(["mean", "std"]).round(5)
print(summary.to_string())

tpe = results.loc[results.sampler == "tpe", "full_data_cv"]
rnd = results.loc[results.sampler == "random", "full_data_cv"]
diff = tpe.mean() - rnd.mean()
spread = max(tpe.std(ddof=1), rnd.std(ddof=1))
se = np.sqrt(tpe.var(ddof=1) / len(tpe) + rnd.var(ddof=1) / len(rnd))
print(f"\nfull-data CV: TPE {tpe.mean():.5f} vs random {rnd.mean():.5f}, difference {diff:+.5f}")
print(f"seed-to-seed std: TPE {tpe.std(ddof=1):.5f}, random {rnd.std(ddof=1):.5f}; standard error of the difference {se:.5f}")
if abs(diff) > spread:
    print(f"VERDICT: the difference ({diff:+.5f}) exceeds the seed-to-seed spread ({spread:.5f}): "
          f"{'TPE' if diff > 0 else 'random search'} is better in this experiment.")
else:
    print(f"VERDICT: the difference ({diff:+.5f}) is within the seed-to-seed spread ({spread:.5f}): "
          "no evidence either sampler is better here.")
print(f"(with only {len(SAMPLER_SEEDS)} seeds per sampler, even a difference above the spread is weak evidence)")
