# CS 166 Group 6 Project

Using XGBoost to build a classification model to determine whether a given Twitter USer is a bot or human (not a bot)

Create dataset.py & create_dataset.ipynb are used to add more datapoints to the csv we originally obtained from kaggle -> created "output.csv" as our dataset that we input into the model

## Running

```python predict.py {TWITTER_ID}```
## Training

trained_model.json is the currently trained model.

Modifying train_model.py and running it will output a new model to be used by predict.py
