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

import random

DEVICE = "cuda"

class Collate:
    def __init__(self):
        self.text_encoder = TextEncoder(DEVICE)
        self.visual_encoder = VisualEncoder(DEVICE)


    def collate_fn(self, filepaths):
        images = []
        tags = []

        for filepath in filepaths:
            image, image_tags, orig_filepath = load_cached_image(filepath)
            images.append(image[0])
            tags.append(image_tags)

        targets = torch.zeros(len(filepaths), 512)
        for i, image_tags in enumerate(tags):
            if 0 < len(image_tags):
                tag = random.choice(image_tags)
                target = self.text_encoder.encode(tag)
            else:
                data = {"pixel_values": torch.tensor([ images[i] ])}
                input_image = BatchFeature(data, tensor_type="pt")
                target = self.visual_encoder.encode(input_image).detach()

            targets[i] = target[0]

        targets = torch.tensor(targets).to(DEVICE)
            
        data = {"pixel_values": torch.tensor(images)}
        images = BatchFeature(data, tensor_type="pt").to(DEVICE)

        return images, targets

def train():
    filepaths = get_cached_files()

    collate = Collate()

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    for i in tqdm(range(10), desc="Epochs"):
        dataloader = DataLoader(filepaths, batch_size = 70, collate_fn=collate.collate_fn, shuffle=True)
        for batch_images, targets in dataloader:
            outputs = model(**batch_images)

            pooled_outputs = outputs.image_embeds

            loss = ce_loss(pooled_outputs, targets)
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model, "trained_clip_visual_model.pt")
