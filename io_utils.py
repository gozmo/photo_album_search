import os
import json


ANNOTATION_DIR = "data/annotations"

def __init_directory_structure():
    if not os.path.isdir(ANNOTATION_DIR):
        os.makedirs(ANNOTATION_DIR)

def load_annotations(label):
    path = f'{ANNOTATION_DIR}/{label}.json'
    print(path)
    if not os.path.isfile(path):
        print("is_empty")
        return {}

    with open(path, "r") as f:
        dictionary = json.load(f)
    new_dictionary = {int(k):v for k,v in dictionary.items()}
    return new_dictionary



def save_annotations(label, annotations):
    __init_directory_structure()

    saved_annotations = load_annotations(label)
    saved_annotations.update(annotations)

    path = f'{ANNOTATION_DIR}/{label}.json'
    with open(path, "w") as f:
        json.dump(saved_annotations, f)

def list_labels():
    path = f'{ANNOTATION_DIR}'
    filenames = os.listdir(path)
    labels = [os.path.splitext(filename)[0] for filename in filenames]
    return labels
