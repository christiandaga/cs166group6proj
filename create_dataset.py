import numpy as np
import pandas as pd
import tweepy
import config

FILENAME = "twitter_human_bots_dataset.csv"

#load kaggle dataset into a dataframe
pd.read_csv(FILENAME)

#configure twitter api client
access_token = config.access_token
access_token_secret = config.access_token_secret
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#obtain fields from accounts using api client

#insert profile data points into corresponding rows

#remove any junk rows

#export csv