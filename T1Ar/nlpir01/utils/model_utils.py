import tensorflow as tf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
from os import path

from utils.text_sequencer import Sequencer
from utils.global_parameters import RESOURCES_PATH, RESULTS_PATH, RESULTS_PER_CLAIM
from utils.global_parameters import SEQ_LEN, CLAIM_SEQ_LEN, EMBEDDING_SIZE
from utils.global_parameters import TEXT_COLUMN, TEXT_COLUMN2, LABEL_COLUMN
from utils.global_parameters import NUM_WORDS, BATCH_SIZE, EPOCHS

def run_model_binary(model, x_train, y_train, x_test, y_test,
    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):

    history = model.fit(x_train, y_train,
                        batch_size, epochs, validation_split=0.2, verbose=verbose)
    loss = None
    accuracy = None
    if y_test is not None:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    raw_predictions = model.predict(x_test)
    class_predictions = [class_of(x) for x in raw_predictions]
    history_dict = history.history

    return loss, accuracy, class_predictions, history_dict, raw_predictions


def class_of(binary_raw_value):
    if binary_raw_value < 0.5:
        return 0
    else:
        return 1


def export_results(test_df, predictions, results_path, results_file_prefix, run_id):
    test_df["prediction"] = predictions
    results_df = test_df.sort_values(by=["topic_id", "prediction"], ascending=False)
    with open(join(results_path, results_file_prefix + run_id), "w") as results_file:
        last_topic = ""
        for _, row in results_df.iterrows():
            if last_topic != row["topic_id"]:
                rank = 1
            if rank <= RESULTS_PER_CLAIM or RESULTS_PER_CLAIM == 0:
                results_file.write("{}\t{}\t{}\t{:.15f}\t{}\n".format(row["topic_id"], rank, row["tweet_id"],
                    row["prediction"], run_id))
            rank = rank + 1
            last_topic = row["topic_id"]


def export_results_to_evaluate(test_df, predictions, results_path, results_file_prefix, run_id):
    test_df["prediction"] = predictions
    results_df = test_df.sort_values(by=["topic_id", "prediction"], ascending=False)
    last_topic = ""
    for _, row in results_df.iterrows():
        if last_topic != row["topic_id"]:
            rank = 1
            results_file = open(join(results_path, results_file_prefix + row["topic_id"]), "w")
        if rank <= RESULTS_PER_CLAIM or RESULTS_PER_CLAIM == 0:
            results_file.write("{}\t{}\t{}\t{:.15f}\t{}\n".format(row["topic_id"], rank, row["tweet_id"],
                row["prediction"], run_id))
        rank = rank + 1
        last_topic = row["topic_id"]


def get_sequences_from_dataset(train_df, test_df):
    sequencer = Sequencer(train_df[TEXT_COLUMN], NUM_WORDS, tokenizer="simple", lang="ar")
    word_index = sequencer.word_index

    print(f"Found {sequencer.unique_word_count} unique tokens.")

    train_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN], SEQ_LEN)
    test_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN], SEQ_LEN)

    return train_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_df[LABEL_COLUMN].values, word_index


def get_sequences_from_dataset_with_claim(train_df, test_df):
    sequencer = Sequencer(train_df[TEXT_COLUMN], NUM_WORDS, tokenizer="simple", lang="ar")
    word_index = sequencer.word_index

    print(f"Found {sequencer.unique_word_count} unique tokens.")

    train_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN], SEQ_LEN)
    test_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN], SEQ_LEN)

    train_claim_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN2], CLAIM_SEQ_LEN)
    test_claim_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN2], CLAIM_SEQ_LEN)

    return train_seqs, train_claim_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_claim_seqs, test_df[LABEL_COLUMN].values, word_index


def get_sequences_from_dataset_with_claim_rel(train_df, test_df, clef_submission=0):
    sequencer = Sequencer(train_df[TEXT_COLUMN], NUM_WORDS, tokenizer="simple", lang="ar")
    word_index = sequencer.word_index

    print(f"Found {sequencer.unique_word_count} unique tokens.")

    train_0_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN], SEQ_LEN)
    test_0_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN], SEQ_LEN)

    rel_train_1_seqs = sequencer.fit_on_text(train_df["rel_text_1"], SEQ_LEN)
    rel_test_1_seqs = sequencer.fit_on_text(test_df["rel_text_1"], SEQ_LEN)
    rel_train_2_seqs = sequencer.fit_on_text(train_df["rel_text_2"], SEQ_LEN)
    rel_test_2_seqs = sequencer.fit_on_text(test_df["rel_text_2"], SEQ_LEN)
    rel_train_3_seqs = sequencer.fit_on_text(train_df["rel_text_3"], SEQ_LEN)
    rel_test_3_seqs = sequencer.fit_on_text(test_df["rel_text_3"], SEQ_LEN)

    train_seqs = np.concatenate((train_0_seqs, rel_train_1_seqs, rel_train_2_seqs, rel_train_3_seqs), axis=1)
    test_seqs = np.concatenate((test_0_seqs, rel_test_1_seqs, rel_test_2_seqs, rel_test_3_seqs), axis=1)

    train_claim_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN2], CLAIM_SEQ_LEN)
    test_claim_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN2], CLAIM_SEQ_LEN)
    if clef_submission:
        return train_seqs, train_claim_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_claim_seqs, None, word_index
    else:
        return train_seqs, train_claim_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_claim_seqs, test_df[LABEL_COLUMN].values, word_index


def get_embedding_layer(pretrained_embedding_path, word_index):
    pretrained_np_embedding_path = pretrained_embedding_path.split(".")[0] + ".npy"

    if path.exists(pretrained_np_embedding_path):
        embedding_matrix = np.load(pretrained_np_embedding_path)
    else:
        print("Loading embeddings...")
        embeddings_index = {}
        embeddings_file = open(pretrained_embedding_path)
        for line in embeddings_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        embeddings_file.close()

        embedding_matrix = np.zeros((len(word_index) + 1, coefs.shape[0]))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        np.save(pretrained_np_embedding_path, embedding_matrix)
        print("Done!")

    return embedding_matrix


def tfidf_transform(train_df, test_df, text_column):
    tiv = TfidfVectorizer(ngram_range=(1, 1))
    train_tiv_fit = tiv.fit(train_df[text_column])
    train_tiv_fit_transform = train_tiv_fit.transform(train_df[text_column])
    train_x = train_tiv_fit_transform.toarray()
    test_x = train_tiv_fit.transform(test_df[text_column]).toarray()
    return train_x, test_x


def get_tfidf_with_claim(train_df, test_df):
    train_x, test_x = tfidf_transform(train_df, test_df, TEXT_COLUMN)
    train_claim_x, test_claim_x = tfidf_transform(train_df, test_df, TEXT_COLUMN2)
    train_y = train_df.pop(LABEL_COLUMN).values
    test_y = test_df.pop(LABEL_COLUMN).values
    return train_x, train_claim_x, train_y, test_x, test_claim_x, test_y


def get_tfidf_with_claim_rel(train_df, test_df, clef_submission=0):
    train_0_x, test_0_x = tfidf_transform(train_df, test_df, TEXT_COLUMN)
    rel_train_1_x, rel_test_1_x = tfidf_transform(train_df, test_df, "rel_text_1")
    rel_train_2_x, rel_test_2_x = tfidf_transform(train_df, test_df, "rel_text_2")
    rel_train_3_x, rel_test_3_x = tfidf_transform(train_df, test_df, "rel_text_3")

    train_x = np.concatenate((train_0_x, rel_train_1_x, rel_train_2_x, rel_train_3_x), axis=1)
    test_x = np.concatenate((test_0_x, rel_test_1_x, rel_test_2_x, rel_test_3_x), axis=1)

    train_claim_x, test_claim_x = tfidf_transform(train_df, test_df, TEXT_COLUMN2)

    train_y = train_df.pop(LABEL_COLUMN).values
    test_y = None
    if clef_submission == 0:
        test_y = test_df.pop(LABEL_COLUMN).values
    return train_x, train_claim_x, train_y, test_x, test_claim_x, test_y
