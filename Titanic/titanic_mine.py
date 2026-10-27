import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
train_file = "/home/ubuntu/projects/kaggle_notebooks/Titanic/data/train.csv"
test_file = "/home/ubuntu/projects/kaggle_notebooks/Titanic/data/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
train_df.info()

print(train_df.head(5))
print(test_df.head(5))
print(test_df[test_df['PassengerId']==925])

# 1. Separate features and target
X = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X_test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
X_val['Age'] = X_val['Age'].fillna(X_train['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X_train['Age'].median())
X_test['Fare'] = X_test['Fare'].fillna(X_train['Fare'].median())


# Scale numeric columns
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Convert object columns to pandas category dtype
cat_cols = ['Sex', 'Embarked']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')



model = xgb.XGBClassifier(enable_categorical=True, objective='binary:logistic', eval_metric='logloss')

model.fit(X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            verbose=True)

model.save_model("xgb_model_mine.json")

preds = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds})
submission.to_csv("submission_xgb_mine.csv", index=False)

