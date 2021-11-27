from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd

#Preprossesing (temp)
df = pd.read_csv('output.csv', sep='\t')

df.head()
df.drop(['id', 'name', 'screen_name', 'time_zone'], axis=1, inplace=True)
df.columns = df.columns.str.replace(' ', '_')
df = df.drop(columns=df.columns[0])
df.dtypes

X = df.drop('account_type', axis=1).copy()
Y = df['account_type'].copy()

#Training & Exporting
seed = 45

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, stratify=Y)
model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, n_estimators=1000, learning_rate=0.01, seed=seed)
model.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

model.save_model("trained_model.json")