import csv
import os
import pyconll
from io import BytesIO
from itertools import compress
from pathlib import Path
from typing import Union, List, Dict
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import random
import shutil
import string
from typing import Union
import tweepy

def download_unzip(url_zip: str,
                   dir_extract: str) -> str:
    """Download and unzip a ZIP archive to folder.

    Loads a ZIP file from URL and extracts all of the files to a 
    given folder. Does not save the ZIP file itself.

    Args:
        url_zip (str): URL to ZIP file.
        dir_extract (str): Directory where files are extracted.

    Returns:
        str: a message telling, if the archive was succesfully
        extracted. Obviously the files in the ZIP archive are
        extracted to the desired directory as a side-effect.
    """
    
    print(f'Reading {url_zip}')
    with urlopen(url_zip) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'

def download_angrytweets() -> str:
    """Download AngryTweets

    Downloads the 'AngryTweets' data set annotated for 
    sentiment analysis, developed and hosted by 
    [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#angrytweets).
                          
    Returns:
        str: a message telling, if the archive was in fact 
        succesfully extracted. Obviously the AngryTweets dataset 
        is downloaded and unzipped as a side-effect.
    """
    # set to default directory if nothing else has been provided by user.
    dir = os.path.join(str(Path.home()), '.danlp')

    return download_unzip(url_zip = 'http://danlp-downloads.alexandra.dk/datasets/game_tweets.zip',
                          dir_extract = dir)

#### ... a lot of copy pasta from 'danlp' follows ####
def _lookup_tweets(tweet_ids, api):
    import tweepy
    full_tweets = []
    tweet_count = len(tweet_ids)
    try:
        for i in range(int(tweet_count/100)+1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            full_tweets.extend(
                api.statuses_lookup(id_=tweet_ids[i * 100:end_loc], tweet_mode='extended', trim_user=True)
            )
        return full_tweets
    except tweepy.TweepError:
        print("Failed fetching tweets")

import sys
import tweepy
def _construct_twitter_api_connection():
    if not('TWITTER_CONSUMER_KEY' in os.environ
           and 'TWITTER_CONSUMER_SECRET' in os.environ
           and 'TWITTER_ACCESS_TOKEN' in os.environ
           and 'TWITTER_ACCESS_SECRET' in os.environ):
        sys.exit("The Twitter API keys was not found."
              "\nTo download tweets you need to set the following environment "
              "variables: \n- 'TWITTER_CONSUMER_KEY'\n- 'TWITTER_CONSUMER_SECRET'"
              "\n- 'TWITTER_ACCESS_TOKEN'\n- 'TWITTER_ACCESS_SECRET' "
              "\n\nThe keys can be obtained from "
              "https://developer.twitter.com")

    twitter_consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
    twitter_consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
    twitter_access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
    twitter_access_secret = os.environ.get('TWITTER_ACCESS_SECRET')

    auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
    auth.set_access_token(twitter_access_token, twitter_access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
    except tweepy.TweepError:
        sys.exit("Could not establish connection to the Twitter API, have you provieded the correct keys?")

    return api

def get_angrytweets(force: bool = False) -> dict:
    """Load DaNE data split.

    Args:
        force (bool): force download of tweets from
            Twitter API.

    Returns:
        DataFrame: tweets annotated with polarity.
    """
    dir = os.path.join(str(Path.home()), '.danlp')
    if not os.path.isdir(dir):
        download_angrytweets()
  
    # check if tweets have actually been collected before.
    dataset_path = os.path.join(dir, 'game_tweets_collected.csv')
    if os.path.isfile(dataset_path):
        if not force:
            print(f"tweets read from file {dataset_path}")
            return pd.read_csv(dataset_path)
    
    file_path = os.path.join(dir, 'game_tweets.csv')
    if not os.path.isfile(file_path):
        download_angrytweets()
            
    df = pd.read_csv(file_path)

    twitter_api = _construct_twitter_api_connection()

    twitter_ids = list(df['twitterid'])
    
    full_t = _lookup_tweets(twitter_ids, twitter_api)
    tweet_texts = [[tweet.id, tweet.full_text] for tweet in full_t]
    tweet_ids, t_texts = list(zip(*tweet_texts))
    tweet_texts_df = pd.DataFrame({'twitterid': tweet_ids, 'text': t_texts})

    resulting_df = pd.merge(df, tweet_texts_df)
    
    resulting_df.to_csv(dataset_path, index=False)
    
    print("Downloaded {} out of {} tweets".format(len(full_t), len(twitter_ids)))

    return resulting_df




    
