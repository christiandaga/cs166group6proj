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
seed = 45
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, stratify=Y)

# params = {
#     'objective': 'binary:logistic',
#     'max_depth': 6,
#     'n_estimators': 1000,
#     'learning_rate': 0.01,
#     'seed': 45}

model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, n_estimators=1000, learning_rate=0.01, seed=seed)
model.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# @harkai99 (human)
# test = [[33, 574, 712]]
# test_df = pd.DataFrame(test, columns=['followers_count', 'friends_count', 'statuses_count'])
# test_pred = model.predict(test_df)
# print(test_pred[0])

# # @a_quilt_bot (bot)
# test = [[2983, 2, 135500]]
# test_df = pd.DataFrame(test, columns=['followers_count', 'friends_count', 'statuses_count'])
# test_pred = model.predict(test_df)
# print(test_pred[0])

model.save_model("trained_model.json")

# import matplotlib.pyplot as plt
# import os

# os.environ["PATH"] += os.pathsep + 'C:/Users/jaras/anaconda3/Library/bin/graphviz/'

# xgb.plot_tree(model)
# plt.show()

# filename = 'qwerasdf.png'
# gvz = xgb.to_graphviz(model, num_trees=model.best_iteration, rankdir='UT')
# _, file_extension = os.path.splitext(filename)
# format = file_extension.strip('.').lower()
# data = gvz.pipe(format=format)
# full_filename = filename
# with open(full_filename, 'wb') as f:
#     f.write(data)
