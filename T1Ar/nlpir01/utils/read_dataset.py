# -*- coding: utf-8 -*-

import gzip
import codecs
import json
import pandas as pd
import sys
import os
from os.path import join
import networkx as nx
import csv

from utils.tweets_graph import TweetsGraph
from utils.global_parameters import TWEETS_PATH, LABELS_PATH, TOPICS_PATH
from utils.global_parameters import TEST_TWEETS_PATH, TEST_TOPICS_PATH
from utils.global_parameters import RESULTS_PATH, GOLD_PATH

sys.path.append(os.path.abspath("."))


def read_tweets(tweets_path):
    tweets_array = []
    with gzip.open(tweets_path, "rt", encoding="utf-8") as tweets:
        for tweet in tweets:
            decoded_data = codecs.decode(tweet.encode(), "utf-8")
            tweet_json = json.loads(decoded_data)
            tweets_array.append(tweet_json)
    tweets_df = pd.DataFrame(tweets_array)
    return tweets_df


def read_labels():
    labels = pd.read_csv(LABELS_PATH, sep="\t", header=None)
    labels_df = pd.DataFrame(labels)
    labels_df.columns = ["topicID", "tweetID", "label"]
    return labels_df


def read_topics(topics_path):
    topics_description = []
    for line in open(topics_path, "r", encoding="utf-8"):
        description = json.loads(line)
        topics_description.append(description)
    topics_df = pd.DataFrame(topics_description)
    topics_df.columns = ["topicID", "title", "description"]
    return topics_df


def get_combined_dataset():
    tweets_df = read_tweets(TWEETS_PATH)
    labels_df = read_labels()
    topics_df = read_topics(TOPICS_PATH)
    labels_tweets_df = pd.merge(labels_df, tweets_df, left_on="tweetID", right_on="id")
    all_df = pd.merge(topics_df, labels_tweets_df, left_on="topicID", right_on="topicID_x")
    return all_df


def get_text_only_dataset():
    df = get_combined_dataset()
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweetID"]
        row_dict["topic_id"] = row["topicID"]
        row_dict["label"] = row["label"]
        row_dict["claim_text"] = row["title"] + " " + row["description"]
        row_dict["text"] = row["full_text"]
        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def get_combined_test_dataset():
    tweets_df = read_tweets(TEST_TWEETS_PATH)
    topics_df = read_topics(TEST_TOPICS_PATH)
    all_df = pd.merge(topics_df, tweets_df, left_on="topicID", right_on="topicID")
    return all_df


def get_text_only_test_dataset():
    df = get_combined_test_dataset()
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweetID"]
        row_dict["topic_id"] = row["topicID"]
        row_dict["claim_text"] = row["title"] + " " + row["description"]
        row_dict["text"] = row["full_text"]
        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def get_id_text_df(df):
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweetID"]
        row_dict["text"] = row["full_text"]
        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def get_text_only_dataset_with_graph_features():  
    tweets_graph = TweetsGraph(TWEETS_PATH)

    df = get_combined_dataset()
    result = []
    for _, row in df.iterrows():

        feature_nodes = tweets_graph.get_feature_nodes(row["tweetID"])

        row_dict = {}
        row_dict["tweet_id"] = row["tweetID"]
        row_dict["topic_id"] = row["topicID"]
        row_dict["label"] = row["label"]
        row_dict["claim_text"] = row["title"] + " " + row["description"]
        row_dict["text"] = row["full_text"]

        row_dict["rel_tweet_id_1"] = 0
        row_dict["rel_text_1"] = ""
        row_dict["rel_tweet_id_2"] = 0
        row_dict["rel_text_2"] = ""
        row_dict["rel_tweet_id_3"] = 0
        row_dict["rel_text_3"] = ""
        if len(feature_nodes) > 0:
            row_dict["rel_tweet_id_1"] = feature_nodes[0]
            row_dict["rel_text_1"] = df.loc[df["tweetID"] == feature_nodes[0]]["full_text"].values[0]
        if len(feature_nodes) > 1:
            row_dict["rel_tweet_id_2"] = feature_nodes[1]
            row_dict["rel_text_2"] = df.loc[df["tweetID"] == feature_nodes[1]]["full_text"].values[0]
        if len(feature_nodes) > 2:
            row_dict["rel_tweet_id_3"] = feature_nodes[2]
            row_dict["rel_text_3"] = df.loc[df["tweetID"] == feature_nodes[2]]["full_text"].values[0]

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def get_text_only_test_dataset_with_graph_features():
    tweets_graph = TweetsGraph(TEST_TWEETS_PATH)

    df = get_combined_test_dataset()
    result = []
    for _, row in df.iterrows():

        feature_nodes = tweets_graph.get_feature_nodes(row["id"])

        row_dict = {}
        row_dict["tweet_id"] = row["id"]
        row_dict["topic_id"] = row["topicID"]
        row_dict["claim_text"] = row["title"] + " " + row["description"]
        row_dict["text"] = row["full_text"]

        row_dict["rel_tweet_id_1"] = 0
        row_dict["rel_text_1"] = ""
        row_dict["rel_tweet_id_2"] = 0
        row_dict["rel_text_2"] = ""
        row_dict["rel_tweet_id_3"] = 0
        row_dict["rel_text_3"] = ""
        if len(feature_nodes) > 0:
            row_dict["rel_tweet_id_1"] = feature_nodes[0]
            row_dict["rel_text_1"] = df.loc[df["id"] == feature_nodes[0]]["full_text"].values[0]
        if len(feature_nodes) > 1:
            row_dict["rel_tweet_id_2"] = feature_nodes[1]
            row_dict["rel_text_2"] = df.loc[df["id"] == feature_nodes[1]]["full_text"].values[0]
        if len(feature_nodes) > 2:
            row_dict["rel_tweet_id_3"] = feature_nodes[2]
            row_dict["rel_text_3"] = df.loc[df["id"] == feature_nodes[2]]["full_text"].values[0]

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def clean_dataset(df):
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweet_id"]
        row_dict["topic_id"] = row["topic_id"]
        row_dict["label"] = row["label"]
        row_dict["claim_text"] = row["claim_text"]
        row_dict["text"] = row["text"]
        row_dict["rel_tweet_id_1"] = row["rel_tweet_id_1"]
        row_dict["rel_text_1"] = row["rel_text_1"]
        row_dict["rel_tweet_id_2"] = row["rel_tweet_id_2"]
        row_dict["rel_text_2"] = row["rel_text_2"]
        row_dict["rel_tweet_id_3"] = row["rel_tweet_id_3"]
        row_dict["rel_text_3"] = row["rel_text_3"]

        if row["rel_tweet_id_1"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_1"]]) == 0:
                row_dict["rel_text_1"] = ""
        if row["rel_tweet_id_2"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_2"]]) == 0:
                row_dict["rel_text_2"] = ""
        if row["rel_tweet_id_3"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_3"]]) == 0:
                row_dict["rel_text_3"] = ""

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def clean_test_dataset(df):
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweet_id"]
        row_dict["topic_id"] = row["topic_id"]
        row_dict["claim_text"] = row["claim_text"]
        row_dict["text"] = row["text"]
        row_dict["rel_tweet_id_1"] = row["rel_tweet_id_1"]
        row_dict["rel_text_1"] = row["rel_text_1"]
        row_dict["rel_tweet_id_2"] = row["rel_tweet_id_2"]
        row_dict["rel_text_2"] = row["rel_text_2"]
        row_dict["rel_tweet_id_3"] = row["rel_tweet_id_3"]
        row_dict["rel_text_3"] = row["rel_text_3"]

        if row["rel_tweet_id_1"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_1"]]) == 0:
                row_dict["rel_text_1"] = ""
        if row["rel_tweet_id_2"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_2"]]) == 0:
                row_dict["rel_text_2"] = ""
        if row["rel_tweet_id_3"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_3"]]) == 0:
                row_dict["rel_text_3"] = ""

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def split_labels():
    with open(LABELS_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        last_topic = ""
        csv_splitted_file = None
        fieldnames = ["topic", "tweet_id", "label"]
        for row in csv_reader:
            if row[0] != last_topic:
                csv_splitted_file = open(join(GOLD_PATH, row[0] + "-Train-T1-Labels.txt"), mode="w")
            writer = csv.DictWriter(csv_splitted_file, fieldnames=fieldnames, delimiter='\t')
            writer.writerow({"topic": row[0], "tweet_id": row[1], "label": row[2]})

            last_topic  = row[0]
