import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_hub as hub
import numpy as np
import pandas as pd
import scipy.spatial
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


from datetime import datetime
from tqdm import tqdm
from os import path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

sys.path.append(os.path.abspath("."))

from lib.logger import logger
from evaluate import extract_metrics, METRICS, METRICS_DEPTH, MAX_DEPTH
from format_checker import run_checks
from trectools import TrecRun, TrecQrel, TrecEval

from utils.global_parameters import TWEETS_TRAIN_PATH, GOLD_LABELS_TRAIN_PATH, TWEETS_DEV_PATH, GOLD_LABELS_DEV_PATH
from utils.global_parameters import TRAIN_DF_FILE, DEV_DF_FILE, VCLAIMS_PATH,RESULTS_FILE_PATH, LABEL_COLUMN, PREDICT_FILE_COLUMNS
from utils.global_parameters import TEST_DF_FILE, TWEETS_TEST_PATH, PREDICT_FILE, RESULT_FILE, PREDICT_SUB_FILE, TRAINING_PNG_FILE
from utils.models import get_ffnn_model
from utils.model_utils import fit_model, evaluate_model_binary
from utils.text_features2 import get_text_features


def build_subplot(model_label, history_dict):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.title(model_label)
    plt.plot(epochs, loss, "r-", label="Training loss")
    plt.plot(epochs, acc, "y-", label="Training acc")
    plt.plot(epochs, val_loss, "g-", label="Validation loss")
    plt.plot(epochs, val_acc, "b-", label="Validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()


def build_dataset_embeddings(tweets_path, vclaims_path, gold_label_path, complete):
    vclaims = pd.read_csv(vclaims_path, sep='\t', index_col=0)
    tweets = pd.read_csv(tweets_path, sep='\t', index_col=0)
    gold_labels = pd.read_csv(gold_label_path, sep='\t', index_col=0, names=["tweet_id", "0", "vclaim_id", "relevance"])
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    tweets_count, vclaims_count, labels_count = len(tweets), len(vclaims), len(gold_labels)

    labels = {}
    for tweet_id, label in tqdm(gold_labels.iterrows(), total=labels_count):
        labels[tweet_id] = label[1]

    claims_embedding = {}
    titles_embedding = {}
    for vclaim_id, vclaim in tqdm(vclaims.iterrows(), total=vclaims_count):
        vclaim_embedding = embed([vclaim[0], vclaim[1]])
        claims_embedding[vclaim_id] = vclaim_embedding[0]
        titles_embedding[vclaim_id] = vclaim_embedding[1]

    data = []
    for tweet_id, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        tweet_embedding = embed([tweet.tweet_content])
        zero_count = 0
        for vclaim_id, _ in vclaims.iterrows():

            if complete == 1:
                add_instance = True
            else:
                zero_count += 1
                add_instance = False
                if labels[tweet_id] == vclaim_id:
                    add_instance = True
                else:
                    if zero_count < 15:
                        add_instance = True
                    else:
                        add_instance = False

            if add_instance:
                data_instance = {}
                data_instance["tweet_id"] = tweet_id
                data_instance["vclaim_id"] = vclaim_id
                data_instance["tweet_embedding"] = tweet_embedding[0]
                data_instance["claim_embedding"] = claims_embedding[vclaim_id]
                data_instance["title_embedding"] = titles_embedding[vclaim_id]
                data_instance["score"] = 2.1 - scipy.spatial.distance.cosine(tweet_embedding[0], claims_embedding[vclaim_id])
                if labels[tweet_id] == vclaim_id:
                    data_instance["label"] = 1
                else:
                    data_instance["label"] = 0

                data.append(data_instance)

    data_df = pd.DataFrame(data)
    return data_df


def build_dataset_text_features(tweets_path, vclaims_path, gold_label_path, complete, title_features=0, all_features=0):
    vclaims = pd.read_csv(vclaims_path, sep='\t', index_col=0)
    tweets = pd.read_csv(tweets_path, sep='\t', index_col=0)

    tweets_count, vclaims_count = len(tweets), len(vclaims)

    labels = {}
    if gold_label_path is not None:
        gold_labels = pd.read_csv(gold_label_path, sep='\t', index_col=0, names=["tweet_id", "0", "vclaim_id", "relevance"])
        labels_count = len(gold_labels)
        for tweet_id, label in tqdm(gold_labels.iterrows(), total=labels_count):
            labels[tweet_id] = label[1]

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    claims_embedding = {}
    titles_embedding = {}
    for vclaim_id, vclaim in tqdm(vclaims.iterrows(), total=vclaims_count):
        vclaim_embedding = embed([vclaim[0], vclaim[1]])
        claims_embedding[vclaim_id] = vclaim_embedding[0]
        titles_embedding[vclaim_id] = vclaim_embedding[1]

    data = []
    for tweet_id, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        tweet_embedding = embed([tweet.tweet_content])
        zero_count = 0

        tw_char_count, tw_word_count, tw_type_token_ratio, \
        tw_average_word_count, tw_n_letter_word_count, tw_noun_count, \
        tw_verb_count, tw_main_verb_count, tw_modal_verb_count, \
        tw_modifiers_count, tw_adjectives_count, tw_adverbs_count, \
        tw_content_word_diversity, tw_content_tag_diversity, \
        tw_causation_word_count, tw_exclusive_word_count, \
        tw_negation_word_count, tw_negative_emotions_word_count, \
        tw_cognitive_word_count, tw_first_person_word_count, \
        tw_second_person_word_count, tw_third_person_word_count = get_text_features(tweet.tweet_content)

        for vclaim_id, vclaim in vclaims.iterrows():

            if complete == 1:
                add_instance = True
            else:
                zero_count += 1
                add_instance = False
                if labels[tweet_id] == vclaim_id:
                    add_instance = True
                else:
                    if zero_count < 20:
                        add_instance = True
                    else:
                        add_instance = False

            if add_instance:
                c_char_count, c_word_count, c_type_token_ratio, \
                c_average_word_count, c_n_letter_word_count, c_noun_count, \
                c_verb_count, c_main_verb_count, c_modal_verb_count, \
                c_modifiers_count, c_adjectives_count, c_adverbs_count, \
                c_content_word_diversity, c_content_tag_diversity, \
                c_causation_word_count, c_exclusive_word_count, \
                c_negation_word_count, c_negative_emotions_word_count, \
                c_cognitive_word_count, c_first_person_word_count, \
                c_second_person_word_count, c_third_person_word_count = get_text_features(vclaim[0])

                t_char_count, t_word_count, t_type_token_ratio, \
                t_average_word_count, t_n_letter_word_count, t_noun_count, \
                t_verb_count, t_main_verb_count, t_modal_verb_count, \
                t_modifiers_count, t_adjectives_count, t_adverbs_count, \
                t_content_word_diversity, t_content_tag_diversity, \
                t_causation_word_count, t_exclusive_word_count, \
                t_negation_word_count, t_negative_emotions_word_count, \
                t_cognitive_word_count, t_first_person_word_count, \
                t_second_person_word_count, t_third_person_word_count = get_text_features(vclaim[1])


                data_instance = {}
                data_instance["tweet_id"] = tweet_id
                data_instance["vclaim_id"] = vclaim_id

                data_instance["score_claim"] = scipy.spatial.distance.cosine(tweet_embedding[0], claims_embedding[vclaim_id])
                data_instance["c_type_token_ratio"] = c_type_token_ratio - tw_type_token_ratio
                data_instance["c_average_word_count"] = c_average_word_count - tw_average_word_count
                data_instance["c_noun_count"] = c_noun_count - tw_noun_count
                data_instance["c_verb_count"] = c_verb_count - tw_verb_count
                data_instance["c_content_word_diversity"] = c_content_word_diversity - tw_content_word_diversity
                data_instance["c_content_tag_diversity"] = c_content_tag_diversity - tw_content_tag_diversity

                if title_features == 1:
                    data_instance["score_title"] = scipy.spatial.distance.cosine(tweet_embedding[0], titles_embedding[vclaim_id])
                    data_instance["t_type_token_ratio"] = t_type_token_ratio - tw_type_token_ratio
                    data_instance["t_average_word_count"] = t_average_word_count - tw_average_word_count
                    data_instance["t_noun_count"] =  t_noun_count - tw_noun_count
                    data_instance["t_verb_count"] = t_verb_count - tw_verb_count
                    data_instance["t_content_word_diversity"] = t_content_word_diversity - tw_content_word_diversity
                    data_instance["t_content_tag_diversity"] = t_content_tag_diversity - tw_content_tag_diversity


                if all_features == 1:
                    data_instance["c_char_count"] = c_char_count - tw_char_count
                    data_instance["c_word_count"] = c_word_count - tw_word_count
                    data_instance["c_n_letter_word_count"] = c_n_letter_word_count - tw_n_letter_word_count
                    data_instance["c_main_verb_count"] = c_main_verb_count - tw_main_verb_count
                    data_instance["c_modal_verb_count"] = c_modal_verb_count - tw_modal_verb_count
                    data_instance["c_modifiers_count"] = c_modifiers_count - tw_modifiers_count
                    data_instance["c_adjectives_count"] = c_adjectives_count - tw_adjectives_count
                    data_instance["c_adverbs_count"] = c_adverbs_count - tw_adverbs_count
                    data_instance["c_causation_word_count"] = c_causation_word_count - tw_causation_word_count
                    data_instance["c_exclusive_word_count"] = c_exclusive_word_count - tw_exclusive_word_count
                    data_instance["c_negation_word_count"] = c_negation_word_count - tw_negation_word_count
                    data_instance["c_negative_emotions_word_count"] = c_negative_emotions_word_count - tw_negative_emotions_word_count
                    data_instance["c_cognitive_word_count"] = c_cognitive_word_count - tw_cognitive_word_count
                    data_instance["c_first_person_word_count"] = c_first_person_word_count - tw_first_person_word_count
                    data_instance["c_second_person_word_count"] = c_second_person_word_count - tw_second_person_word_count
                    data_instance["c_third_person_word_count"] = c_third_person_word_count - tw_third_person_word_count
                    data_instance["t_char_count"] = t_char_count - tw_char_count
                    data_instance["t_word_count"] = t_word_count - tw_word_count
                    data_instance["t_n_letter_word_count"] = t_n_letter_word_count - tw_n_letter_word_count
                    data_instance["t_main_verb_count"] = t_main_verb_count - tw_main_verb_count
                    data_instance["t_modal_verb_count"] = t_modal_verb_count - tw_modal_verb_count
                    data_instance["t_modifiers_count"] = t_modifiers_count - tw_modifiers_count
                    data_instance["t_adjectives_count"] = t_adjectives_count - tw_adjectives_count
                    data_instance["t_adverbs_count"] = t_adverbs_count - tw_adverbs_count
                    data_instance["t_causation_word_count"] = t_causation_word_count - tw_causation_word_count
                    data_instance["t_exclusive_word_count"] = t_exclusive_word_count - tw_exclusive_word_count
                    data_instance["t_negation_word_count"] = t_negation_word_count - tw_negation_word_count
                    data_instance["t_negative_emotions_word_count"] = t_negative_emotions_word_count - tw_negative_emotions_word_count
                    data_instance["t_cognitive_word_count"] = t_cognitive_word_count - tw_cognitive_word_count
                    data_instance["t_first_person_word_count"] = t_first_person_word_count - tw_first_person_word_count
                    data_instance["t_second_person_word_count"] = t_second_person_word_count - tw_second_person_word_count
                    data_instance["t_third_person_word_count"] = t_third_person_word_count - tw_third_person_word_count

                if gold_label_path is not None:
                    if labels[tweet_id] == vclaim_id:
                        data_instance["label"] = 1
                    else:
                        data_instance["label"] = 0

                data.append(data_instance)

    data_df = pd.DataFrame(data)
    return data_df


def get_score(tweet, vclaims, embed, embed_claims_claim, embed_claims_title, model):
    embed_tweet = embed([tweet])
    results = []
    for i, _ in vclaims.iterrows():

        x_test0 = np.hstack((embed_tweet[0], embed_claims_claim[i]))
        x_test0 = x_test0.reshape(1, x_test0.shape[0])
        raw_predictions = model.predict(x_test0)

        result = {}
        result["_id"] = i
        result["_score"] = raw_predictions[0][0] + 1
        results.append(result)
    df = pd.DataFrame(results)
    df["id"] = df._id.astype('int32').values
    df = df.set_index("id")
    return df._score


def get_scores(tweets, vclaims, embed, model):

    tweets_count, vclaims_count = len(tweets), len(vclaims)

    embed_claims_claim = {}
    embed_claims_title = {}
    for i, vclaim in tqdm(vclaims.iterrows(), total=vclaims_count):
        embed_vclaims = embed([vclaim[0], vclaim[1]])
        embed_claims_claim[i] = embed_vclaims[0]
        embed_claims_title[i] = embed_vclaims[1]

    scores = {}
    logger.info(f"Geting RM5 scores for {tweets_count} tweets and {vclaims_count} vclaims")
    for i, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        score = get_score(tweet.tweet_content, vclaims, embed, embed_claims_claim, embed_claims_title, model)
        scores[i] = score
    return scores


def format_scores(scores):
    formatted_scores = []
    for tweet_id, s in scores.items():
        for vclaim_id, score in s.items():
            row = (str(tweet_id), 'Q0', str(vclaim_id), '1', str(score), 'nlpir01')
            formatted_scores.append(row)
    formatted_scores_df = pd.DataFrame(formatted_scores, columns=PREDICT_FILE_COLUMNS)
    return formatted_scores_df


def make_predictions(model):
    vclaims = pd.read_csv(VCLAIMS_PATH, sep='\t', index_col=0)
    tweets = pd.read_csv(TWEETS_DEV_PATH, sep='\t', index_col=0)
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    scores = get_scores(tweets, vclaims, embed, model)

    formatted_scores = format_scores(scores)
    formatted_scores.to_csv(PREDICT_FILE, sep='\t', index=False, header=False)
    logger.info(f"Saved scores from the model in file: {PREDICT_FILE}")


def export_scores(result, predict_file, max_rows):
    curr_row = 0
    curr_tweet = -1
    formatted_scores = []
    for _, row in result.iterrows():
        if curr_tweet != int(row.tweet_id):
            curr_row = 0
        curr_row = curr_row + 1
        if curr_row <= max_rows:
            row2 = (str(int(row.tweet_id)), 'Q0', str(int(row.vclaim_id)), '1', "{:0.20f}".format(row.score), 'nlpir01')
            formatted_scores.append(row2)
        curr_tweet = int(row.tweet_id)
    formatted_scores_df = pd.DataFrame(formatted_scores, columns=PREDICT_FILE_COLUMNS)

    formatted_scores_df.to_csv(predict_file, sep='\t', index=False, header=False)
    logger.info(f"Saved scores from the model in file: {predict_file}")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--title_features", "-tf", type=int, default=0, choices=[0, 1],
        help="0 = use claim-tweet features, 1 = use claim-tweet and claim title-tweet features.")
    parser.add_argument("--all_features", "-a", type=int, default=0, choices=[0, 1],
        help="0 = do not use the rest of the claim-tweet and claim title-tweet features, 1 = use the rest of the claim-tweet and claim title-tweet features.")
    args = parser.parse_args()  

    starting_time = datetime.now()

    if path.exists(TRAIN_DF_FILE):
        train0_df = pd.read_pickle(TRAIN_DF_FILE)
    else:
        train0_df = build_dataset_text_features(
            TWEETS_TRAIN_PATH, VCLAIMS_PATH, GOLD_LABELS_TRAIN_PATH, 0, 
            title_features=args.title_features, all_features=args.all_features)
        train0_df.to_pickle(TRAIN_DF_FILE)

    print("Original train dataset")
    print(train0_df.label.value_counts())

    train_df = train0_df.sample(frac=0.80, random_state=0)
    test_df = train0_df.drop(train_df.index)

    print("\nTrain dataset")
    print(train_df.label.value_counts())
    print("\nTest dataset")
    print(test_df.label.value_counts())

    if path.exists(DEV_DF_FILE):
        dev_df = pd.read_pickle(DEV_DF_FILE)
    else:
        dev_df = build_dataset_text_features(
            TWEETS_DEV_PATH, VCLAIMS_PATH, GOLD_LABELS_DEV_PATH, 1, 
            title_features=args.title_features, all_features=args.all_features)
        dev_df.to_pickle(DEV_DF_FILE)

    print("Dev dataset")
    print(dev_df.label.value_counts())
    print()


    print("\nTest dataset for submission")
    print()
    if path.exists(TEST_DF_FILE):
        test_sub_df = pd.read_pickle(TEST_DF_FILE)
    else:
        test_sub_df = build_dataset_text_features(
            TWEETS_TEST_PATH, VCLAIMS_PATH, None, 1, 
            title_features=args.title_features, all_features=args.all_features)
        test_sub_df.to_pickle(TEST_DF_FILE)

    y_train = train_df[LABEL_COLUMN].values
    x_train = train_df.drop(["tweet_id", "vclaim_id", "label"], axis=1).to_numpy()

    y_test = test_df[LABEL_COLUMN].values
    x_test = test_df.drop(["tweet_id", "vclaim_id", "label"], axis=1).to_numpy()

    input_length = x_train.shape[1]
    print("Input size:", input_length)

    model = get_ffnn_model(input_length, 1000, activation="elu", optimizer="adam", verbose=0)
    trained_model, history_dict = fit_model(model, x_train, y_train, epochs=50, batch_size=128, verbose=0)
    loss, accuracy, class_predictions, raw_predictions = evaluate_model_binary(trained_model, x_test, y_test, verbose=0)

    print(f"test loss: {loss} test accuracy: {accuracy}\n")
    print(classification_report(y_test, class_predictions, zero_division=0) + "\n")

    plt.figure(figsize=(12,9))

    plt.subplot(111)
    build_subplot("model5", history_dict)

    plt.tight_layout()
    plt.savefig(TRAINING_PNG_FILE)

    y_dev = dev_df[LABEL_COLUMN].values
    x_dev = dev_df.drop(["tweet_id", "vclaim_id", "label"], axis=1).to_numpy()


    loss, accuracy, class_predictions, raw_predictions = evaluate_model_binary(trained_model, x_dev, y_dev, verbose=0)
    print(f"dev loss: {loss} dev accuracy: {accuracy}\n")
    print(classification_report(y_dev, class_predictions, zero_division=0) + "\n")

    results = dev_df.assign(score=pd.Series(raw_predictions.reshape(1, -1)[0]).values).sort_values(["tweet_id", "score"], ascending=[True, False])
    export_scores(results, PREDICT_FILE, 1000)


    x_test_sub = test_sub_df.drop(["tweet_id", "vclaim_id"], axis=1).to_numpy()
    test_sub_raw_predictions = trained_model.predict(x_test_sub)
    results_sub = test_sub_df.assign(score=pd.Series(test_sub_raw_predictions.reshape(1, -1)[0]).values).sort_values(["tweet_id", "score"], ascending=[True, False])
    export_scores(results_sub, PREDICT_SUB_FILE, 1000)

    format_check_passed = run_checks(PREDICT_SUB_FILE)
    if format_check_passed:
        print("Submission predictions format OK")
    else:
        print("Submission predictions format ERROR")


    format_check_passed = run_checks(PREDICT_FILE)
    if format_check_passed:
        gold_labels = TrecQrel(GOLD_LABELS_DEV_PATH)
        prediction = TrecRun(PREDICT_FILE)

        results = TrecEval(prediction, gold_labels)
        metrics = extract_metrics(results, METRICS, METRICS_DEPTH)

        metrics.loc[:, '@depth'] = metrics.loc[:, '@depth'].astype(str)
        metrics.loc[:, '@depth'] = metrics.loc[:, '@depth'].replace(str(MAX_DEPTH), 'all')
        metrics.to_csv(RESULT_FILE, sep='\t', index=False)
        logger.info(f'Saved results to file: {RESULT_FILE}')
        print(metrics.to_string(index=False))

    ending_time = datetime.now()
    print("Total time:", ending_time - starting_time)

    print("Done!!!")