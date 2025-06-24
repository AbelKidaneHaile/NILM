# functions to manupulate JSON files
import json
import os


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, output_path, indent=4):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def file_exists(path=None):
    return os.path.isfile(path)


def list_directories_as_json(folder_path, output_file=None):
    directories = [
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    result = {"folder_path": os.path.abspath(folder_path), "directories": directories}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Directory list saved to {output_file}")
    else:
        print(json.dumps(result, indent=4))


def build_directory_tree(path, use_absolute=True, base_path=None):
    if base_path is None:
        base_path = os.getcwd()

    tree = {
        "name": os.path.basename(path),
        "path": (
            os.path.abspath(path) if use_absolute else os.path.relpath(path, base_path)
        ),
        "subdirectories": [],
    }

    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    subtree = build_directory_tree(entry.path, use_absolute, base_path)
                    tree["subdirectories"].append(subtree)
    except PermissionError:
        pass  # Skip folders we can't access

    return tree


def save_directory_tree_json(root_folder, output_file, use_absolute=True):
    tree = build_directory_tree(root_folder, use_absolute=use_absolute)
    with open(output_file, "w") as f:
        json.dump(tree, f, indent=5)
    print(f"Directory tree saved to {output_file}")
