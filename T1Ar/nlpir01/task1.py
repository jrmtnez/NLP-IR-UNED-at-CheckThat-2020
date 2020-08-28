import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.metrics import classification_report
from datetime import datetime
from os.path import join, dirname

sys.path.append(os.path.abspath("."))

from scorer.main import get_mean_values

from utils.path_utils import get_file_with_path_list, prepare_folders_and_files
from utils.models import get_ffnn_emb_model, get_cnn_emb_model, get_lstm_emb_model
from utils.models import get_bilstm_emb_model, get_ffnn_model
from utils.read_dataset import get_text_only_dataset, clean_dataset, get_text_only_dataset_with_graph_features
from utils.read_dataset import get_text_only_test_dataset, clean_test_dataset
from utils.read_dataset import get_text_only_test_dataset_with_graph_features, split_labels
from utils.text_sequencer import Sequencer
from utils.model_utils import get_embedding_layer, get_sequences_from_dataset_with_claim, get_sequences_from_dataset_with_claim_rel
from utils.model_utils import run_model_binary, export_results, export_results_to_evaluate, get_tfidf_with_claim, get_tfidf_with_claim_rel
from utils.global_parameters import TEXT_COLUMN, TEXT_COLUMN2, LABEL_COLUMN, EMBEDDINGS_FILE, NUM_WORDS
from utils.global_parameters import RESULTS_PER_CLAIM, RESOURCES_PATH, RESULTS_PATH
from utils.global_parameters import RESULTS_TO_EVALUATE_PATH, GOLD_PATH, RESULTS_FILE_PREFIX


MODEL1_PREFIX_LABEL = "Model 1: Dense"
MODEL2_PREFIX_LABEL = "Model 2: CNN"
MODEL3_PREFIX_LABEL = "Model 3: LSTM"
MODEL4_PREFIX_LABEL = "Model 4: Bi-LSTM"
MODEL5_PREFIX_LABEL = "Model 5: Dense TF-IDF"
MODEL6_PREFIX_LABEL = "Model 6: "
MODEL1_SHORT_LABEL = "nlpir01-run1"
MODEL2_SHORT_LABEL = "nlpir01-run2"
MODEL3_SHORT_LABEL = "nlpir01-run3"
MODEL4_SHORT_LABEL = "nlpir01-run4"
MODEL5_SHORT_LABEL = "nlpir01-run5"
MODEL6_SHORT_LABEL = "nlpir01-run6"


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


def get_data(graph_features=0, use_embeddings=1, clef_submission=0):

    word_index = None

    if graph_features == 0:
        df = get_text_only_dataset()

        train_df = df.sample(frac=0.80, random_state=0)
        test_df = df.drop(train_df.index)

        if use_embeddings == 1:
            x_text_train, x_claim_train, y_train, x_text_test, x_claim_test, y_test, word_index = get_sequences_from_dataset_with_claim(
                train_df, test_df)
        else:
            x_text_train, x_claim_train, y_train, x_text_test, x_claim_test, y_test = get_tfidf_with_claim(train_df, test_df)
    else:
        if clef_submission == 0:
            df = get_text_only_dataset_with_graph_features()

            train_df = df.sample(frac=0.80, random_state=0)
            test_df = df.drop(train_df.index)
        else:
            train_df = get_text_only_dataset_with_graph_features()
            test_df = get_text_only_test_dataset_with_graph_features()

        train_df = clean_dataset(train_df)

        if clef_submission == 0:
            test_df = clean_dataset(test_df)
        else:
            test_df = clean_test_dataset(test_df)

        if use_embeddings == 1:
            x_text_train, x_claim_train, y_train, x_text_test, x_claim_test, y_test, word_index = get_sequences_from_dataset_with_claim_rel(
                train_df, test_df, clef_submission=clef_submission)
        else:
            x_text_train, x_claim_train, y_train, x_text_test, x_claim_test, y_test = get_tfidf_with_claim_rel(
                train_df, test_df, clef_submission=clef_submission)

    x_train = np.concatenate((x_text_train, x_claim_train), axis=1)
    x_test = np.concatenate((x_text_test, x_claim_test), axis=1)
    input_length = x_train[:].shape[1]  # [:] to avoid pylint unsuscriptable warning

    if use_embeddings == 1:
        print("Embeddings input:")
    else:
        print("TF-IDF input:")
    print("\tShape of text tensor:", x_text_train.shape)
    print("\tShape of claim text tensor:", x_claim_train.shape)
    print("\tShape of concatenated x_train tensor:", x_train.shape)
    print("\tShape of class tensor:", y_train.shape)
    print("\tInput lenght:", input_length)

    return input_length, x_train, y_train, x_test, y_test, test_df, word_index


def run_task1(models_to_run, embeddings=0, graph_features=0, verbose=0, clef_submission=0):
    
    prepare_folders_and_files()

    if clef_submission == 0:
        split_labels()

    if embeddings == 1:
        label_embbedings = "glove"
    else:
        label_embbedings = "autoemb"

    if graph_features == 1:
        label_graph = "graph"
    else:
        label_graph = "nograph"

    model1_label = MODEL1_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model2_label = MODEL2_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model3_label = MODEL3_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model4_label = MODEL4_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model5_label = MODEL5_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model6_label = MODEL6_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph

    input_length, x_train, y_train, x_test, y_test, test_df, word_index = get_data(graph_features, use_embeddings=1, clef_submission=clef_submission)
    input_length2, x_train2, y_train2, x_test2, y_test2, test_df, _ = get_data(graph_features, use_embeddings=0, clef_submission=clef_submission)

    embedding_matrix = None
    if embeddings == 1:
        embedding_matrix = get_embedding_layer(
            os.path.join(RESOURCES_PATH, EMBEDDINGS_FILE), word_index)

    if 1 in models_to_run:
        print("Running", model1_label)
        model1 = get_ffnn_emb_model(embedding_matrix, word_index, input_length, 2000,
                    activation="relu", optimizer="adam", verbose=verbose)
        loss1, accuracy1, class_predictions1, history_dict1, raw_predictions1 = run_model_binary(
            model1, x_train, y_train, x_test, y_test, epochs=50, batch_size=8, verbose=verbose)
        export_results(test_df, raw_predictions1, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL1_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions1, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL1_SHORT_LABEL)
            overall_precisions, avg_precision1, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_1 = overall_precisions[5]

    if 2 in models_to_run:
        print("Running", model2_label)
        model2 = get_cnn_emb_model(embedding_matrix, word_index, input_length, 100,
                    activation="relu", optimizer="adam", verbose=verbose)
        loss2, accuracy2, class_predictions2, history_dict2, raw_predictions2 = run_model_binary(
            model2, x_train, y_train, x_test, y_test, epochs=25, batch_size=8, verbose=verbose)
        export_results(test_df, raw_predictions2, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL2_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions2, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL2_SHORT_LABEL)
            overall_precisions, avg_precision2, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_2 = overall_precisions[5]

    if 3 in models_to_run:
        print("Running", model3_label)
        model3 = get_lstm_emb_model(embedding_matrix, word_index, input_length,
                    dropout=0.1, recurrent_dropout=0, optimizer="adam", verbose=verbose)
        loss3, accuracy3, class_predictions3, history_dict3, raw_predictions3 = run_model_binary(
            model3, x_train, y_train, x_test, y_test, epochs=5, batch_size=32, verbose=verbose)
        export_results(test_df, raw_predictions3, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL3_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions3, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL3_SHORT_LABEL)
            overall_precisions, avg_precision3, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_3 = overall_precisions[5]

    if 4 in models_to_run:
        print("Running", model4_label)
        model4 = get_bilstm_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size=10,
                    dropout=0.1, recurrent_dropout=0, activation="tanh", optimizer="nadam", verbose=verbose)
        loss4, accuracy4, class_predictions4, history_dict4, raw_predictions4 = run_model_binary(
            model4, x_train, y_train, x_test, y_test, epochs=5, batch_size=32, verbose=verbose)
        export_results(test_df, raw_predictions4, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL4_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions4, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL4_SHORT_LABEL)
            overall_precisions, avg_precision4, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_4 = overall_precisions[5]

    if 5 in models_to_run:
        print("Running", model5_label)
        model5 = get_ffnn_model(input_length2, 500,
                    dropout=0.4, activation="relu", optimizer="adam", verbose=verbose)
        loss5, accuracy5, class_predictions5, history_dict5, raw_predictions5 = run_model_binary(
            model5, x_train2, y_train2, x_test2, y_test2, epochs=10, batch_size=8, verbose=verbose)
        export_results(test_df, raw_predictions5, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL5_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions5, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL5_SHORT_LABEL)
            overall_precisions, avg_precision5, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_5 = overall_precisions[5]

    if 6 in models_to_run:
        print("Running", model6_label)
        model6 = None
        loss6, accuracy6, class_predictions6, history_dict6, raw_predictions6 = run_model_binary(
            model6, x_train, y_train, x_test, y_test, epochs=10, batch_size=8, verbose=verbose)
        export_results(test_df, raw_predictions6, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL6_SHORT_LABEL)
        if clef_submission == 0:
            export_results_to_evaluate(test_df, raw_predictions6, RESULTS_TO_EVALUATE_PATH, RESULTS_FILE_PREFIX, MODEL6_SHORT_LABEL)
            overall_precisions, avg_precision6, _, _ = get_mean_values(get_file_with_path_list(GOLD_PATH), get_file_with_path_list(RESULTS_TO_EVALUATE_PATH))
            avgp30_6 = overall_precisions[5]


    if clef_submission == 0:
        with open(join(RESULTS_PATH, RESULTS_FILE_PREFIX + "results_" + label_embbedings + "_" + label_graph + ".txt"), "w") as results_file:
            if 1 in models_to_run:
                results_file.write(model1_label + "\n")
                results_file.write(f"test loss: {loss1} test accuracy: {accuracy1}\n")
                results_file.write(f"avg precision: {avg_precision1}\n")
                results_file.write(f"avg p@30: {avgp30_1}\n")
                results_file.write(classification_report(y_test, class_predictions1) + "\n")

            if 2 in models_to_run:
                results_file.write(model2_label + "\n")
                results_file.write(f"test loss: {loss2} test accuracy: {accuracy2}\n")
                results_file.write(f"avg precision: {avg_precision2}\n")
                results_file.write(f"avg p@30: {avgp30_2}\n")
                results_file.write(classification_report(y_test, class_predictions2) + "\n")

            if 3 in models_to_run:
                results_file.write(model3_label + "\n")
                results_file.write(f"test loss: {loss3} test accuracy: {accuracy3}\n")
                results_file.write(f"avg precision: {avg_precision3}\n")
                results_file.write(f"avg p@30: {avgp30_3}\n")
                results_file.write(classification_report(y_test, class_predictions3) + "\n")

            if 4 in models_to_run:
                results_file.write(model4_label + "\n")
                results_file.write(f"test loss: {loss4} test accuracy: {accuracy4}\n")
                results_file.write(f"avg precision: {avg_precision4}\n")
                results_file.write(f"avg p@30: {avgp30_4}\n")
                results_file.write(classification_report(y_test, class_predictions4) + "\n")

            if 5 in models_to_run:
                results_file.write(model5_label + "\n")
                results_file.write(f"test loss: {loss5} test accuracy: {accuracy5}\n")
                results_file.write(f"avg precision: {avg_precision5}\n")
                results_file.write(f"avg p@30: {avgp30_5}\n")(
                results_file.write(classification_report(y_test, class_predictions5) + "\n"))

            if 6 in models_to_run:
                results_file.write(model6_label + "\n")
                results_file.write(f"test loss: {loss6} test accuracy: {accuracy6}\n")
                results_file.write(f"avg precision: {avg_precision6}\n")
                results_file.write(f"avg p@30: {avgp30_6}\n")
                results_file.write(classification_report(y_test, class_predictions6) + "\n")


    if clef_submission == 0:
        if 1 in models_to_run:
            print("\n" + model1_label)
            print(f"test loss: {loss1} test accuracy: {accuracy1}")
            print(f"avg precision: {avg_precision1}")
            print(f"avg p@30: {avgp30_1}")
            print(classification_report(y_test, class_predictions1))

        if 2 in models_to_run:
            print("\n" + model2_label)
            print(f"test loss: {loss2} test accuracy: {accuracy2}")
            print(f"avg precision: {avg_precision2}")
            print(f"avg p@30: {avgp30_2}")
            print(classification_report(y_test, class_predictions2))

        if 3 in models_to_run:
            print("\n" + model3_label)
            print(f"test loss: {loss3} test accuracy: {accuracy3}")
            print(f"avg precision: {avg_precision3}")
            print(f"avg p@30: {avgp30_3}")
            print(classification_report(y_test, class_predictions3))

        if 4 in models_to_run:
            print("\n" + model4_label)
            print(f"test loss: {loss4} test accuracy: {accuracy4}")
            print(f"avg precision: {avg_precision4}")
            print(f"avg p@30: {avgp30_4}")
            print(classification_report(y_test, class_predictions4))

        if 5 in models_to_run:
            print("\n" + model5_label)
            print(f"test loss: {loss5} test accuracy: {accuracy5}")
            print(f"avg precision: {avg_precision5}")
            print(f"avg p@30: {avgp30_5}")
            print(classification_report(y_test, class_predictions5))

        if 6 in models_to_run:
            print("\n" + model6_label)
            print(f"test loss: {loss6} test accuracy: {accuracy6}")
            print(f"avg precision: {avg_precision6}")
            print(f"avg p@30: {avgp30_6}")
            print(classification_report(y_test, class_predictions6))


    if clef_submission == 0:
        plt.figure(figsize=(12, 9))

        if 1 in models_to_run:
            plt.subplot(331)
            build_subplot(model1_label, history_dict1)

        if 2 in models_to_run:
            plt.subplot(332)
            build_subplot(model2_label, history_dict2)

        if 3 in models_to_run:
            plt.subplot(333)
            build_subplot(model3_label, history_dict3)

        if 4 in models_to_run:
            plt.subplot(334)
            build_subplot(model4_label, history_dict4)

        if 5 in models_to_run:
            plt.subplot(335)
            build_subplot(model5_label, history_dict5)

        if 6 in models_to_run:
            plt.subplot(336)
            build_subplot(model6_label, history_dict6)

        plt.tight_layout()
        plt.savefig(join(RESULTS_PATH, RESULTS_FILE_PREFIX + "results_" + label_embbedings + "_" + label_graph + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", "-s", type=int, default=0, choices=[0, 1],
        help='0 = run on training dataset, 1 = run on test dataset for submission to CheckThat!')
    parser.add_argument("--models", "-m", type=list, default=[1, 2, 4],
        help='List of models to run. All models = [1, 2, 3, 4, 5]')
    parser.add_argument("--embeddings", "-e", type=int, default=1, choices=[0, 1],
        help='When submission = 1, 0 = auto-generated embeddings, 1 = Glove embeddings.')
    parser.add_argument("--graph", "-g", type=int, default=1, choices=[0, 1],
        help='When submission = 1, 0 = do not use graph, 1 = use graph.')
    args = parser.parse_args()  

    starting_time = datetime.now()

    if args.submission == 0:
        run_task1(args.models, embeddings=1, graph_features=1, clef_submission=args.submission)
        run_task1(args.models, embeddings=1, graph_features=0, clef_submission=args.submission)
        run_task1(args.models, embeddings=0, graph_features=1, clef_submission=args.submission)
        run_task1(args.models, embeddings=0, graph_features=0, clef_submission=args.submission)
    else:
        run_task1(args.models, embeddings=args.embeddings, graph_features=args.graph, clef_submission=args.submission)

    ending_time = datetime.now()

    print("Total time:", ending_time - starting_time)
    print("Done!!!")
