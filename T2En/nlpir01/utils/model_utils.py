import numpy as np

from os.path import join
from os import path

from utils.global_parameters import BATCH_SIZE, EPOCHS


def run_model_binary(model, x_train, y_train, x_test, y_test,
    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):
    trained_model, history_dict = fit_model(model, x_train, y_train, batch_size, epochs, verbose)
    loss, accuracy, class_predictions, raw_predictions = evaluate_model_binary(trained_model, x_test, y_test, verbose)
    return loss, accuracy, class_predictions, history_dict, raw_predictions


def fit_model(model, x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):
    history = model.fit(x_train, y_train, batch_size, epochs, validation_split=0.2, verbose=verbose)
    history_dict = history.history
    return model, history_dict


def evaluate_model_binary(model, x_test, y_test, verbose=0):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    raw_predictions = model.predict(x_test)
    class_predictions = [class_of(x) for x in raw_predictions]
    return loss, accuracy, class_predictions, raw_predictions


def class_of(binary_raw_value):
    if binary_raw_value < 0.5:
        return 0
    else:
        return 1

