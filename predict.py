import xgboost as xgb
import pandas as pd
import tweepy
import config
import sys

#configure token stuff
access_token = config.access_token
access_token_secret = config.access_token_secret
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)


try:
    user_id = sys.argv[1]
    twitter_user = api.get_user(params={"user_id":user_id})
    current_user = {
        "followers_count": twitter_user["followers_count"],
        "friends_count": twitter_user["friends_count"],
        "statuses_count": twitter_user["statuses_count"],
    }
    print(current_user)
    columns = ['followers_count', 'friends_count', 'statuses_count']
    index = ['a', 'b', 'c']
    dataframe = pd.DataFrame(current_user, columns=columns, index=index)
    model = xgb.XGBClassifier(objective="binary:logistic", seed=42)
    model.load_model('trained_model.json')
    pred = model.predict(dataframe)
    print(twitter_user["screen_name"] + ': ' + pred[0])
except Exception as err:
    print("Error:", err)

