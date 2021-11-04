from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd


# Preprossesing (temp)
df = pd.read_csv('output.csv', sep='\t')
df.head()
df.drop(['id', 'name', 'screen_name', 'time_zone'], axis=1, inplace=True)
df.columns = df.columns.str.replace(' ', '_')
df = df.drop(columns=df.columns[0])
df.dtypes
X = df.drop('account_type', axis=1).copy()
Y = df['account_type'].copy()


# Training and exporting
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, stratify=Y)

model = xgb.XGBClassifier(objective="binary:logistic", seed=seed)
model.fit(X_train, y_train, verbose=True, early_stopping_rounds=50, eval_metric='aucpr', eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# @harkai99 (human)
test = [[33, 574, 712]]
test_df = pd.DataFrame(test, columns=['followers_count', 'friends_count', 'statuses_count'])
test_pred = model.predict(test_df)
print(test_pred[0])

# @a_quilt_bot (bot)
test = [[2983, 2, 135500]]
test_df = pd.DataFrame(test, columns=['followers_count', 'friends_count', 'statuses_count'])
test_pred = model.predict(test_df)
print(test_pred[0])

model.save_model("trained_model.json")
