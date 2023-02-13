import os
import json
import torch

from constants import Directories



def __init_directory_structure():
    if not os.path.isdir(Directories.ANNOTATION):
        os.makedirs(Directories.ANNOTATION)
    if not os.path.isdir(Directories.MODEL_REPO):
        os.makedirs(Directories.MODEL_REPO)

def load_annotations(label):
    path = f'{Directories.ANNOTATION}/{label}.json'
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

    path = f'{Directories.ANNOTATION}/{label}.json'
    with open(path, "w") as f:
        json.dump(saved_annotations, f)

def list_labels():
    path = f'{Directories.ANNOTATION}'
    filenames = os.listdir(path)
    labels = [os.path.splitext(filename)[0] for filename in filenames]
    return labels

def save_model(label, model):
    __init_directory_structure()
    path = f"{Directories.MODEL_REPO}/{label}.pt"
    torch.save(model, path)

def load_model(label):
    path = f"{Directories.MODEL_REPO}/{label}.pt"
    model = torch.load(path)
    return model

def save_threshold(label, threshold):
    path = f"{Directories.MODEL_REPO}/{label}.config"
    content = {"threshold": threshold}
    with open(path, "w") as f:
        json.dump(content, f)

def load_threshold(label):
    path = f"{Directories.MODEL_REPO}/{label}.config"
    with open(path, "r") as f:
        content = json.load(f)

    return content["threshold"]
