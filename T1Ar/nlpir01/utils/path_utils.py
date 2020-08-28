import os
from os.path import join, dirname
from os import listdir

from utils.global_parameters import RESOURCES_PATH, EMBEDDINGS_FILE
from utils.global_parameters import RESULTS_PATH, RESULTS_TO_EVALUATE_PATH, GOLD_PATH


def get_file_with_path_list(path):
    all_files = [join(path, file_name) for file_name in listdir(path)]
    all_files.sort()
    return all_files


def prepare_folders_and_files():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    if not os.path.exists(RESULTS_TO_EVALUATE_PATH):
        os.makedirs(RESULTS_TO_EVALUATE_PATH)

    if not os.path.exists(GOLD_PATH):
        os.makedirs(GOLD_PATH)
    
    embeddings_binary = os.path.join(RESOURCES_PATH, EMBEDDINGS_FILE).split(".")[0] + ".npy"
    if os.path.exists(embeddings_binary):
        os.remove(embeddings_binary)

