from os.path import join, dirname
from os import listdir

def get_file_with_path_list(path):
    all_files = [join(path, file_name) for file_name in listdir(path)]
    all_files.sort()
    return all_files