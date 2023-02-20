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

import random

DEVICE = "cuda"

class Collate:
    def __init__(self):
        self.text_encoder = TextEncoder(DEVICE)
        self.visual_encoder = VisualEncoder(DEVICE)

    def collate_fn_load_images(self, filepath):

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

        targets = torch.zeros(len(filepaths), 512)
        for i, image_tags in enumerate(tags):
            if 0 < len(image_tags):
                sentence = self.__generate_sentence(image_tags)
                target = self.text_encoder.encode(sentence)
            else:
                data = {"pixel_values": torch.tensor([ images[i] ])}
                input_image = BatchFeature(data, tensor_type="pt")
                target = self.visual_encoder.encode(input_image).detach()

            targets[i] = target[0]

        targets = torch.tensor(targets).to(DEVICE)
            
        data = {"pixel_values": torch.tensor(images)}
        images = BatchFeature(data, tensor_type="pt").to(DEVICE)

        return images, targets

    def __generate_sentence(self, tags):
        sentence = "a picture of "
        tag_subsentence = "and".join(tags)

        return sentence + tag_subsentence

def train(load_images=True):
    dataset = []
    collate = Collate()
    if load_images:
        image, image_tags, orig_filepath = load_cached_image(filepath)
        dataset.append( (image, image_tags) )
        collate_fn = collate.collate_fn_images_in_memory
    else:
        dataset = get_cached_files()
        collate_fn = collate.collate_fn_load_images

    batch_size = 70
    lr_warm_up_steps = 200
    gradient_accumulation_steps = 10
    learning_rate = 10e-5
    weight_decay = 0.01
    epochs = 100

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    kk
    schedule = get_constant_schedule_with_warmup(optimizer,
                                      num_warmup_steps=gradient_accumulation_steps)

    for i in tqdm(range(epochs), desc="Epochs"):
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        for batch_images, targets in dataloader:
            outputs = model(**batch_images)

            pooled_outputs = outputs.image_embeds

            pooled_outputs_normalized = torch.nn.functional.normalize(pooled_outputs, dim=0, p=1) 
            targets_normalized = torch.nn.functional.normalize(pooled_outputs, dim=0, p=1) 

            loss = ce_loss(pooled_outputs_normalized, targets_normalized)


            if i % gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                schedule.step()

    torch.save(model, "trained_clip_visual_model.pt")
