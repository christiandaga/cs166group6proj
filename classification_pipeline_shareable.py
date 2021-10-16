import sys
import tweepy
import igraph
import networkx as nx
import pandas as pd
import numpy as np
import csv
import ast
from operator import itemgetter
from karateclub import Graph2Vec
import xgboost as xgb
import config

#Fill in your creds here
access_token = config.access_token
access_token_secret = config.access_token_secret
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


#1. GML (and CSV) creation

# Collects a given user's tweets into a CSV
# Credit to Jacob Moore
class TweetGrabber():
	
    def __init__(self,myApi,sApi,at,sAt):
        self.tweepy = tweepy
        auth = tweepy.OAuthHandler(myApi, sApi)
        auth.set_access_token(at, sAt)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
		
	#Return the string without non ASCII characters
    def strip_non_ascii(self,string):
		
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)  
		
    def user_search(self,user,csv_prefix):
        API_results = self.tweepy.Cursor(self.api.user_timeline,screen_name=user,tweet_mode='extended').items()

        with open(f'{csv_prefix}.csv', 'w', newline='') as csvfile:
            fieldnames = ['tweet_id', 'tweet_text', 'date', 'user_id', 'user_mentions', 'retweet_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for tweet in API_results:
                text = self.strip_non_ascii(tweet.full_text)
                date = tweet.created_at.strftime('%m/%d/%Y')        
                writer.writerow({
								'tweet_id': tweet.id_str,
								'tweet_text': text,
								'date': date,
								'user_id': tweet.user.id_str,
								'user_mentions':tweet.entities['user_mentions'],
                                'retweet_count': tweet.retweet_count
                                })

# Process the created CSV in order to generate edge list
class RetweetParser():
	
    def __init__(self,data,user):
        self.user = user

        edge_list = []
	
        for idx,row in data.iterrows():
            if len(row[4]) > 5:    
                user_account = user
                weight = np.log(row[5] + 1)
                for idx_1, item in enumerate(ast.literal_eval(row[4])):
                    edge_list.append((user_account,item['screen_name'],weight))

                    for idx_2 in range(idx_1+1,len(ast.literal_eval(row[4]))):
                        name_a = ast.literal_eval(row[4])[idx_1]['screen_name']
                        name_b = ast.literal_eval(row[4])[idx_2]['screen_name']

                        edge_list.append((name_a,name_b,weight))
		
        with open(f'{self.user}.csv', 'w', newline='') as csvfile:
            fieldnames = ['user_a', 'user_b', 'log_retweet']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in edge_list:        
                writer.writerow({
								'user_a': row[0],
								'user_b': row[1],
								'log_retweet': row[2]
								})


# Eigenvector centrality measures 'influence' of each node within the graph network
class TweetGraph():
	def __init__(self,edge_list):
		data = pd.read_csv(edge_list).to_records(index=False)
		self.tuple_graph = igraph.Graph.TupleList(data, weights=True, directed=False)
		
	def e_centrality(self):
		vectors = self.tuple_graph.eigenvector_centrality()
		e = {name:cen for cen, name in  zip([v for v in vectors],self.tuple_graph.vs['name'])}
		return sorted(e.items(), key=itemgetter(1),reverse=True)


# Instantiation
t = TweetGrabber(
	myApi = consumer_key,
	sApi = consumer_secret,
	at = access_token,
	sAt = access_token_secret)


# Variable to hold whatever Twitter user is being classified
screen_name = sys.argv[1]


try:
	existing_gml = igraph.read(screen_name + '.gml')
	print(screen_name + '.gml already exists.')
except FileNotFoundError:
    try:
        print("Scanning activity...")
        try:
    		# Collect the user's mentions into a CSV titled with their username
            t.user_search(user=screen_name, csv_prefix=screen_name)
        except:
            print("error writing to csv")
		# Read the created CSV into a pandas DataFrame for input to RetweetParser class
        userFrame = pd.read_csv(screen_name + ".csv")

		# RetweetParser overwrites the first CSV with a weighted edgelist
        r = RetweetParser(userFrame, screen_name)

		# The weighted, undirected iGraph object
        log_graph = TweetGraph(edge_list= screen_name + ".csv")

		# Add 'size' attribute to each vertex based on its Eigencentrality
		# NOTE: multiplying the value by some consistent large number creates a more intuitive
		# plot, viewing-wise, but doesn't impact classification, since this change is applied
		# to all vertices
        for key, value in log_graph.e_centrality():
            log_graph.tuple_graph.vs.find(name=key)['size'] = value*20

		# Save the graph in GML format
        print("Building gml...")
        log_graph.tuple_graph.write_gml(f=screen_name+".gml")

		# Plot the graph for viewing
        # style = {}
        # style["edge_curved"] = False
        # style["vertex_label"] = m_graph.tuple_graph.vs['name']
        # style["vertex_label_size"] = 5
        # plot(m_graph.tuple_graph, **style)

    except:
        print(screen_name + ' graphing failed.')


# 2. GML conversion to Graph2Vec vector

# Believe it or not, the easiest way I found of doing this was to
# now open the GML files in NetworkX instead of iGraph. 

# In order to do so, I first had to insert a line manually labeling 
# each as a multigraph with this very messy chunk of code.
igraph_gml = open(screen_name+".gml", 'r')
lof = igraph_gml.readlines()
igraph_gml.close()
if lof[4]!="multigraph 1":
  lof.insert(4, "multigraph 1\n")
igraph_gml = open(screen_name + '.gml', 'w')
lof = "".join(lof)
igraph_gml.write(lof)
igraph_gml.close()

# Next, read the GML with NetworkX, then convert each
# node from being labeled by name to being labeled by sequential
# integers, since Graph2Vec requires nodes to be labeled this way
print("Creating vector embedding...")
H = nx.read_gml(screen_name + '.gml', label='name')
convertedgraph = nx.convert_node_labels_to_integers(H)

# Instantiate a Graph2Vec embedding model. There are
# a variety of parameters that can be changed when 
# instantiating the model (see the above link to the Karate Club library),
# but I found 64 feature columns and otherwise default parameters
# to provide the best results
embedding_model = Graph2Vec(dimensions=64, min_count=1)

# Now, fit the model to the NetworkX graph, and store the embedding
# in a pandas DataFrame
embedding_model.fit([convertedgraph])
embeddingsframe = pd.DataFrame(embedding_model.get_embedding())


# 3. Use XGBoost classification model to predict user type based on vector,
# and output prediction and plot of user's GML representation

#Load classification model and make a prediction
classification_model = xgb.XGBClassifier(objective="binary:logistic")
classification_model.load_model('graph_classifier_3.json')
print("Predicting...")
pred = classification_model.predict(embeddingsframe, base_margin=[7.75042])

print(screen_name + ': ' + pred[0])



