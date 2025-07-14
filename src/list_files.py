import os


def list_files(folder_path):
    return [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
