from os.path import join, dirname
from os import listdir


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False


def get_file_with_path_list(path):
    all_files = [join(path, file_name) for file_name in listdir(path)]
    all_files.sort()
    return all_files