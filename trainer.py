from PIL.Image import ImagePointHandler
from pymilvus import Collection 
import io_utils
import db_utils
import pudb
import numpy as np

from tqdm import tqdm
import torch
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import LeakyReLU 
from torch.nn import Sigmoid
from torch.nn import Sequential
from torch.utils.data import DataLoader 
from collections import OrderedDict
from sklearn.metrics import f1_score
from image_cache import load_cached_image
from image_cache import get_cached_files
from transformers.image_processing_utils import BatchFeature
from encoder import TextEncoder
from encoder import VisualEncoder
from transformers import CLIPVisionModelWithProjection
from transformers import get_constant_schedule_with_warmup
import db_utils
import random
import logging
import constants
import model_repo

DEVICE = "cuda"

class Collate:
    def __init__(self, model_name):
        self.collection_name = db_utils.model_name_to_collection_name(model_name)
        self.model_name = model_name
        self.text_encoder = TextEncoder(DEVICE, model_name)
        self.visual_encoder = VisualEncoder(DEVICE, model_name)

    def collate_fn_load_images(self, filepaths):

        images = []
        tags = []

        for filepath in filepaths:
            image, image_tags, orig_filepath = load_cached_image(filepath)
            images.append(image[0])
            tags.append(image_tags)

        return self.__collate_fn(images, tags)

    def collate_fn_images_in_memory(self, batch):
        images = [elem[0] for elem in batch]
        tags = [elem[1] for elem in batch]

    def __collate_fn(self, images, tags):

        targets = torch.zeros(len(images), 512)
        for i, image_tags in enumerate(tags):
            if 0 < len(image_tags):
                sentence = self.__generate_sentence(image_tags)
                target = self.text_encoder.encode(sentence)
            else:
                data = {"pixel_values": torch.tensor([ images[i] ])}
                input_image = BatchFeature(data, tensor_type="pt")
                target = self.visual_encoder.encode(input_image).detach()

            targets[i] = target[0]

        targets = targets.clone().detach().to(DEVICE)
            
        data = {"pixel_values": torch.tensor(images)}
        images = BatchFeature(data, tensor_type="pt").to(DEVICE)

        return images, targets

    def __generate_sentence(self, tags):
        sentence = "a picture of "
        tag_subsentence = "and".join(tags)

        return sentence + tag_subsentence

def train(model_name_saved, load_images=False):
    model_name = constants.DEFAULT_MODEL
    dataset = []
    collate = Collate(model_name)
    if load_images:
        logging.info("Read image files and storing them in memory")
        filepaths = get_cached_files()
        for filepath in filepaths:
            image, image_tags, orig_filepath = load_cached_image(filepath)
            dataset.append( (image, image_tags) )
            collate_fn = collate.collate_fn_images_in_memory
    else:
        logging.info("Image files will be read on demand in collate_fn")
        dataset = get_cached_files()
        collate_fn = collate.collate_fn_load_images

    dataset = dataset
    batch_size = 36
    lr_warm_up_steps = 200
    gradient_accumulation_steps = 50
    learning_rate = 10e-8
    weight_decay = 0.01
    epochs = 2

    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(DEVICE)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    schedule = get_constant_schedule_with_warmup(optimizer,
                                      num_warmup_steps=lr_warm_up_steps)

    for i in range(epochs):
        accumulated_loss = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        for batch_images, targets in tqdm(dataloader, desc=f"Epoch: {i}"):
            outputs = model(**batch_images)

            pooled_outputs = outputs.image_embeds

            pooled_outputs_normalized = torch.nn.functional.normalize(pooled_outputs, dim=0, p=1) 
            targets_normalized = torch.nn.functional.normalize(pooled_outputs, dim=0, p=1) 

            loss = ce_loss(pooled_outputs_normalized, targets_normalized)
            loss.backward()
            accumulated_loss += loss.item()

            if i % gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                schedule.step()
        print(accumulated_loss)

    text_path, visual_path = model_repo.get_savepath(model_name_saved) 

    collate.text_encoder.text_model.save_pretrained(text_path)
    model.save_pretrained(visual_path)
