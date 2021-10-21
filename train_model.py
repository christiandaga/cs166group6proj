from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

# Preprossesing (example)
df = pd.read_csv('dataset.csv')
X = df.drop('bot', axis=1).copy()
Y = df['bot'].copy()
X_enc = pd.get_dummies(X, columns=['time_zone'])


# Training and exporting
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X_enc, Y, random_state=seed, stratify=Y)

model = xgb.XGBClassifier(objective="binary:logistic", seed=seed)
model.fit(X_train, y_train, verbose=True, early_stopping_rounds=10)

model.save_model("trained_model.json")