# Twitter-graph-classification
Using Graph2Vec and XGBoost to build a classification model for labeling twitter users

To run classification_pipeline_shareable.py, you'll first need to input your twitter 
developer credentials where instructed on lines 17-22. 

Once done, the script takes one command-line argument at runtime, which is the name of 
the Twitter user being classified, and outputs a string labeling them as either a bot or human.
Formatting is as follows:

python classification_pipeline_shareable.py A_TWITTER_HANDLE
