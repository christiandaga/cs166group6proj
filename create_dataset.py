import numpy as np
import pandas as pd
import tweepy
import config

#configure token stuff
access_token = config.access_token
access_token_secret = config.access_token_secret
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret

#load kaggle dataset into a dataframe
FILENAME = "twitter_human_bots_dataset.csv"
twitter_dataframe = pd.read_csv(FILENAME)
print(twitter_dataframe.head())

#configure twitter api client
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#obtain fields from accounts using api client

#insert profile data points into corresponding rows

#remove any junk rows

#export csv