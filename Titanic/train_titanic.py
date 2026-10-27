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

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_len = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Advanced Feature Engineering
print("Performing Advanced Feature Engineering...")

# Extract Title from Name
df['Title'] = df['Name'].apply(lambda name: re.search(r' ([A-Za-z]+)\.', name).group(1) if re.search(r' ([A-Za-z]+)\.', name) else '')

title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Ms': 'Miss',
    'Lady': 'Royalty',
    'Sir': 'Royalty',
    'Capt': 'Officer',
    'Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Dona': 'Royalty'
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
    if size == 1:
        return 'Alone'
    elif 2 <= size <= 4:
        return 'Small'
    else:
        return 'Large'

df['FamilyType'] = df['FamilySize'].apply(get_family_type)

# Ticket Frequency & Fare per Person
ticket_counts = df['Ticket'].value_counts()
df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

# Log transform fares
df['LogFare'] = np.log1p(df['Fare'])
df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])

# Cabin Features
df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
df['HasCabin'] = (df['Cabin'].notna()).astype(int)

# Is Child indicator (Age < 12 or Title == Master)
df['IsChild'] = ((df['Age'] < 12) | (df['Title'] == 'Master')).astype(int)

# Extract Surname
df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

# Woman-Child-Group (WCG) Feature Construction:
# In Titanic, women and children in the same ticket or family group usually have correlated outcomes.
df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['IsChild'] == 1)).astype(int)

# Construct Ticket / Family Group identifiers
df['Group_ID'] = df['Ticket']

# 3. Stratified K-Fold Setup & WCG Target Encoding inside CV loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def calculate_wcg_survival(df_subset, train_indices, test_indices):
    """
    Calculate Out-Of-Fold Woman-Child Group Survival signal to prevent leakage.
    """
    df_res = df_subset.copy()
    df_res['WCG_Survival'] = 0.5 # Default neutral
    
    # Train subset with known labels
    train_part = df_res.iloc[train_indices]
    
    # Group by Group_ID among women & children in train set
    wcg_groups = train_part[train_part['IsWomanOrChild'] == 1].groupby('Group_ID')['Survived'].agg(['count', 'sum', 'mean'])
    
    # Filter for groups with at least 1 woman/child
    wcg_valid = wcg_groups[wcg_groups['count'] >= 1]
    
    for idx in range(len(df_res)):
        gid = df_res.iloc[idx]['Group_ID']
        is_wc = df_res.iloc[idx]['IsWomanOrChild']
        
        if gid in wcg_valid.index:
            g_count = wcg_valid.loc[gid, 'count']
            g_sum = wcg_valid.loc[gid, 'sum']
            
            # If the current sample is in train set, leave-one-out calculation
            if idx in train_indices and is_wc == 1:
                other_count = g_count - 1
                other_sum = g_sum - df_res.iloc[idx]['Survived']
                if other_count > 0:
                    df_res.iloc[idx, df_res.columns.get_loc('WCG_Survival')] = other_sum / other_count
            else:
                # For test set or adult males, use full group mean from train
                if g_count > 0:
                    df_res.iloc[idx, df_res.columns.get_loc('WCG_Survival')] = g_sum / g_count

    return df_res['WCG_Survival'].values

# Encode Categories
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilyType', 'CabinDeck', 'IsAlone', 'HasCabin', 'IsChild']
num_cols = ['Age', 'Fare', 'FarePerPerson', 'LogFare', 'LogFarePerPerson', 'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize']

X_base = df[num_cols + cat_cols].copy()
X_base['Pclass'] = X_base['Pclass'].astype(str)
X_base['IsAlone'] = X_base['IsAlone'].astype(str)
X_base['HasCabin'] = X_base['HasCabin'].astype(str)
X_base['IsChild'] = X_base['IsChild'].astype(str)

X_encoded = pd.get_dummies(X_base, columns=cat_cols, drop_first=True)

X_train_raw = X_encoded.iloc[:train_len].copy()
y_train = train_df['Survived'].values
X_test_raw = X_encoded.iloc[train_len:].copy()

# Add Out-Of-Fold WCG Feature
wcg_oof = np.zeros(train_len)
wcg_test = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    # Calculate WCG for validation fold using train fold
    wcg_vals = calculate_wcg_survival(df.iloc[:train_len], trn_idx, val_idx)
    wcg_oof[val_idx] = wcg_vals[val_idx]

# For test set, calculate WCG using full train set
full_train_indices = list(range(train_len))
dummy_test_indices = list(range(train_len, len(df)))
wcg_full = calculate_wcg_survival(df, full_train_indices, dummy_test_indices)
wcg_test = wcg_full[train_len:]

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

X_train['WCG_Survival'] = wcg_oof
X_test['WCG_Survival'] = wcg_test

num_cols_all = num_cols + ['WCG_Survival']

# Scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols_all] = scaler.fit_transform(X_train[num_cols_all])
X_test_scaled[num_cols_all] = scaler.transform(X_test[num_cols_all])

# 4. Model Training & Ensembling
models = {
    'LightGBM': lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, random_state=42, eval_metric='logloss'),
    'CatBoost': cb.CatBoostClassifier(iterations=180, learning_rate=0.04, depth=4, random_seed=42, verbose=0),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=4, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_split=4, random_state=42),
    'HistGradBoost': HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=42),
    'SVC': SVC(C=1.5, kernel='rbf', probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=500, random_state=42)
}

print("\n--- 5-Fold Cross Validation Results (with WCG Feature) ---")
oof_preds = {}
test_preds = {}

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
        test_pred_folds += model.predict_proba(X_te)[:, 1] / skf.n_splits
        
    acc = accuracy_score(y_train, (oof >= 0.5).astype(int))
    oof_preds[name] = oof
    test_preds[name] = test_pred_folds
    print(f"{name:<20} Out-of-Fold Accuracy: {acc:.4f}")

# Ensembling & Stacking
oof_df = pd.DataFrame(oof_preds)
test_pred_df = pd.DataFrame(test_preds)

top_models = ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'SVC', 'HistGradBoost']
ensemble_oof = oof_df[top_models].mean(axis=1)
ensemble_acc = accuracy_score(y_train, (ensemble_oof >= 0.5).astype(int))
print(f"\nTop Models Soft Voting Ensemble CV Accuracy: {ensemble_acc:.4f}")

# Stacking Classifier Meta-Learner
meta_learner = LogisticRegression(C=0.5, random_state=42)
meta_learner.fit(oof_df[top_models], y_train)
stacked_oof = meta_learner.predict_proba(oof_df[top_models])[:, 1]
stacked_acc = accuracy_score(y_train, (stacked_oof >= 0.5).astype(int))
print(f"Stacked Meta-Learner CV Accuracy:            {stacked_acc:.4f}")

# Final Predictions
final_test_prob = meta_learner.predict_proba(test_pred_df[top_models])[:, 1]
final_predictions = (final_test_prob >= 0.5).astype(int)

# Create submission
os.makedirs('output', exist_ok=True)
sub = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_predictions
})
sub.to_csv('output/submission.csv', index=False)
print(f"\nSaved updated submission to output/submission.csv")
print(f"Predicted Survived distribution:\n{sub['Survived'].value_counts(normalize=True)}")
