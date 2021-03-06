import pandas as pd
import tweepy
import config

#Twitter users added to output.csv
numNewData = 3000

#Tokens
access_token = config.access_token
access_token_secret = config.access_token_secret
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)

#Load Kaggle dataset into an input dataframe
input_dataframe = pd.read_csv("twitter_human_bots_dataset.csv")

def addData(startIndex, numElements, dataframe):
    for x in range(startIndex, startIndex + numElements):
        try:
            user_id = input_dataframe["id"][x]
            twitter_user = api.get_user(params={"user_id":user_id})
            #Parsing the data
            current_user = {
                "id" : user_id,
                "name": twitter_user["name"],
                "screen_name": twitter_user["screen_name"],
                "followers_count": twitter_user["followers_count"],
                "friends_count": twitter_user["friends_count"],
                "statuses_count": twitter_user["statuses_count"],
                "time_zone": twitter_user["time_zone"],
                "account_type": input_dataframe["account_type"][x]
            }
            print(current_user)
            dataframe = dataframe.append(current_user, ignore_index=True)

            #Saves every 100 datapoints for fallback
            if x % 100 == 0:
                dataframe.to_csv("output.csv", sep='\t')
        except Exception as err:
            print("Error:", err)
    return dataframe

try:
    #Get output from previous fetches
    output_dataframe = pd.read_csv("output.csv", sep='\t', index_col=[0])
    latest_id = output_dataframe["id"].iloc[-1]
    start_index = input_dataframe.index[input_dataframe["id"] == latest_id][0] + 1
    output_dataframe = addData(start_index, numNewData, output_dataframe)
except FileNotFoundError:
    #If no output file create new dataframe
    header_list = ['id', 'name', 'screen_name', 'followers_count', 'friends_count', 'statuses_count', 'time_zone', 'account_type']
    output_dataframe = pd.DataFrame(columns=header_list)
    output_dataframe = addData(0, numNewData, output_dataframe)

#Final csv output file
output_dataframe.to_csv("output.csv", sep='\t')