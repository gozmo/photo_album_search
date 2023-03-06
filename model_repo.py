from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPModel
import os
from constants import DEFAULT_MODEL      
from constants import Directories


def get_savepath(model_name, modality):
    path = f"{Directories.MODEL_REPO}/{model_name}/{modality}"
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_visual_model(model_name):
    if model_name == DEFAULT_MODEL or model_name == "default":
        model = CLIPVisionModelWithProjection.from_pretrained(DEFAULT_MODEL)
    else:
        path = get_savepath(model_name, "clip")
        model = CLIPVisionModelWithProjection.from_pretrained(path)
    return model


def get_text_model(model_name):
    if model_name == DEFAULT_MODEL or model_name == "default":
        model = CLIPTextModelWithProjection.from_pretrained(DEFAULT_MODEL)
    else:
        path = get_savepath(model_name, "clip")
        model = CLIPTextModelWithProjection.from_pretrained(path)
    return model

def get_clip_model(model_name):
    if model_name == DEFAULT_MODEL or model_name == "default":
        model = CLIPTextModelWithProjection.from_pretrained(DEFAULT_MODEL)
    else:
        path = get_savepath(model_name, "clip")
        model = CLIPModel.from_pretrained(path)
    return model

