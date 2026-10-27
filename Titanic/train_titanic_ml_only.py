import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

print("=== Pure ML Pipeline (Strictly No Cheating / No External Labels) ===")

# 1. Load Data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_len = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Feature Engineering
print("Engineering Features...")

# Extract Title
df['Title'] = df['Name'].apply(lambda name: re.search(r' ([A-Za-z]+)\.', name).group(1) if re.search(r' ([A-Za-z]+)\.', name) else '')

title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Lady': 'Royalty',
    'Sir': 'Royalty', 'Capt': 'Officer', 'Countess': 'Royalty',
    'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty'
}
df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

# Impute missing Age by Title & Pclass median
df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())

# Impute missing Fare by Pclass median
df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

# Impute missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Family Size & Alone indicator
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Family Type category
def get_family_type(size):
    if size == 1: return 'Alone'
    elif 2 <= size <= 4: return 'Small'
    else: return 'Large'

df['FamilyType'] = df['FamilySize'].apply(get_family_type)

# Ticket Frequency & Fare Per Person
ticket_counts = df['Ticket'].value_counts()
df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
df['LogFare'] = np.log1p(df['Fare'])
df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])

# Cabin Deck & Indicators
df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
df['HasCabin'] = (df['Cabin'].notna()).astype(int)
df['IsChild'] = ((df['Age'] < 12) | (df['Title'] == 'Master')).astype(int)

# Sex x Pclass Interaction Feature
df['Sex_Pclass'] = df['Sex'] + '_P' + df['Pclass'].astype(str)

# Age x Pclass
df['Age_x_Pclass'] = df['Age'] * df['Pclass']

# Woman-Child Group (WCG) Feature Construction (Strict Out-of-Fold Target Encoding)
df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['IsChild'] == 1)).astype(int)
df['Group_ID'] = df['Ticket']

# 3. Stratified K-Fold Setup (10-Fold CV for Maximum Robustness)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def calculate_wcg_survival(df_subset, train_indices, test_indices):
    """
    Calculate Out-Of-Fold Woman-Child Group Survival signal strictly on training set fold.
    """
    df_res = df_subset.copy()
    df_res['WCG_Survival'] = 0.5
    train_part = df_res.iloc[train_indices]
    
    wcg_groups = train_part[train_part['IsWomanOrChild'] == 1].groupby('Group_ID')['Survived'].agg(['count', 'sum'])
    wcg_valid = wcg_groups[wcg_groups['count'] >= 1]
    
    for idx in range(len(df_res)):
        gid = df_res.iloc[idx]['Group_ID']
        is_wc = df_res.iloc[idx]['IsWomanOrChild']
        
        if gid in wcg_valid.index:
            g_count = wcg_valid.loc[gid, 'count']
            g_sum = wcg_valid.loc[gid, 'sum']
            
            if idx in train_indices and is_wc == 1:
                other_count = g_count - 1
                other_sum = g_sum - df_res.iloc[idx]['Survived']
                if other_count > 0:
                    df_res.iloc[idx, df_res.columns.get_loc('WCG_Survival')] = other_sum / other_count
            else:
                if g_count > 0:
                    df_res.iloc[idx, df_res.columns.get_loc('WCG_Survival')] = g_sum / g_count

    return df_res['WCG_Survival'].values

cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilyType', 'CabinDeck', 'Sex_Pclass', 'IsAlone', 'HasCabin', 'IsChild']
num_cols = ['Age', 'Fare', 'FarePerPerson', 'LogFare', 'LogFarePerPerson', 'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize', 'Age_x_Pclass']

X_base = df[num_cols + cat_cols].copy()
for col in ['Pclass', 'IsAlone', 'HasCabin', 'IsChild']:
    X_base[col] = X_base[col].astype(str)

X_encoded = pd.get_dummies(X_base, columns=cat_cols, drop_first=True)

X_train_raw = X_encoded.iloc[:train_len].copy()
y_train = train_df['Survived'].values
X_test_raw = X_encoded.iloc[train_len:].copy()

# Add Out-Of-Fold WCG Feature for Train set
wcg_oof = np.zeros(train_len)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    wcg_vals = calculate_wcg_survival(df.iloc[:train_len], trn_idx, val_idx)
    wcg_oof[val_idx] = wcg_vals[val_idx]

# Add WCG Feature for Test set (computed using all training data)
wcg_full = calculate_wcg_survival(df, list(range(train_len)), list(range(train_len, len(df))))
wcg_test = wcg_full[train_len:]

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()
X_train['WCG_Survival'] = wcg_oof
X_test['WCG_Survival'] = wcg_test

num_cols_all = num_cols + ['WCG_Survival']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols_all] = scaler.fit_transform(X_train[num_cols_all])
X_test_scaled[num_cols_all] = scaler.transform(X_test[num_cols_all])

# 4. Model Suite & Hyperparameter Selection
models = {
    'CatBoost': cb.CatBoostClassifier(iterations=200, learning_rate=0.03, depth=4, l2_leaf_reg=3, random_seed=42, verbose=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_split=4, max_features='sqrt', random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=250, max_depth=5, min_samples_split=4, max_features='sqrt', random_state=42),
    'HistGradBoost': HistGradientBoostingClassifier(max_iter=120, max_depth=3, l2_regularization=1.0, random_state=42),
    'SVC': SVC(C=1.5, kernel='rbf', probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(C=0.8, max_iter=500, random_state=42)
}

print(f"\n--- {n_splits}-Fold Stratified Cross Validation Results ---")
oof_preds = {}
test_preds = {}
cv_scores = []

for name, model in models.items():
    oof = np.zeros(len(X_train))
    test_pred_folds = np.zeros(len(X_test))
    
    use_scaled = name in ['SVC', 'LogisticRegression']
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
        X_tr_f, y_tr_f = X_tr.iloc[train_idx], y_train[train_idx]
        X_val_f, y_val_f = X_tr.iloc[val_idx], y_train[val_idx]
        
        model.fit(X_tr_f, y_tr_f)
        oof[val_idx] = model.predict_proba(X_val_f)[:, 1]
        test_pred_folds += model.predict_proba(X_te)[:, 1] / n_splits
        
    acc = accuracy_score(y_train, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y_train, oof)
    oof_preds[name] = oof
    test_preds[name] = test_pred_folds
    cv_scores.append({'Model': name, 'OOF Accuracy': acc, 'ROC-AUC': auc})
    print(f"{name:<20} Out-of-Fold Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

# 5. Advanced Ensembling & Stacking
oof_df = pd.DataFrame(oof_preds)
test_pred_df = pd.DataFrame(test_preds)

top_models = ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradBoost', 'SVC']

# Weighted Soft Voting Ensemble
weights = [0.25, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05]
weighted_oof = np.zeros(len(y_train))
weighted_test = np.zeros(len(X_test))

for w, model_name in zip(weights, top_models):
    weighted_oof += w * oof_df[model_name]
    weighted_test += w * test_pred_df[model_name]

weighted_acc = accuracy_score(y_train, (weighted_oof >= 0.5).astype(int))
print(f"\nWeighted Ensemble OOF Accuracy: {weighted_acc:.4f}")

# Stacking Meta-Learner
meta_learner = LogisticRegression(C=0.5, random_state=42)
meta_learner.fit(oof_df[top_models], y_train)
stacked_oof = meta_learner.predict_proba(oof_df[top_models])[:, 1]
stacked_acc = accuracy_score(y_train, (stacked_oof >= 0.5).astype(int))
print(f"Stacked Meta-Learner OOF Accuracy: {stacked_acc:.4f}")

# Generate Pure ML Final Predictions
final_ml_prob = meta_learner.predict_proba(test_pred_df[top_models])[:, 1]
final_ml_preds = (final_ml_prob >= 0.5).astype(int)

# Overwrite submission.csv strictly with pure ML predictions
os.makedirs('output', exist_ok=True)
ml_submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_ml_preds
})

ml_submission.to_csv('output/submission.csv', index=False)
print("\n[SUCCESS] Strictly Pure ML submission written to output/submission.csv")
print(f"Class Distribution:\n{ml_submission['Survived'].value_counts(normalize=True)}")
