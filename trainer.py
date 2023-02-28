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
import logging
import constants
import model_repo
from mlflow import log_metric, log_param, log_artifacts
import mlflow


DEVICE = "cuda"

def __generate_sentence(self, tags):
    sentence = "a picture of "
    tag_subsentence = "and".join(tags)

    return sentence + tag_subsentence

def collate_fn(text_encoder, static_visual_encoder, batch):
    images = [elem[0] for elem in batch]
    tags = [elem[1] for elem in batch]

    targets = torch.zeros(len(images), 512)
    for i, image_tags in enumerate(tags):
        if 0 < len(image_tags):
            sentence = __generate_sentence(image_tags)
            target = text_encoder.encode(sentence)
        else:
            data = {"pixel_values": torch.tensor([ images[i] ])}
            input_image = BatchFeature(data, tensor_type="pt")
            target = static_visual_encoder.encode(input_image).detach()

        targets[i] = target[0]

    targets = targets.clone().detach().to(DEVICE)
        
    data = {"pixel_values": torch.tensor(images)}
    images = BatchFeature(data, tensor_type="pt").to(DEVICE)

    return images, targets


def train(model_name_saved, load_images=False):
    with mlflow.start_run():
        model_name = constants.DEFAULT_MODEL
        dataset = []

        logging.info("Image files will be read on demand in collate_fn")
        dataset = get_cached_files()
        collate_fn = collate.collate_fn_load_images

        dataset = dataset
        params = {
            "batch_size" : 36,
            "lr_warm_up_steps" : 100,
            "gradient_accumulation_steps" : 50,
            "learning_rate" : 1e-8,
            "weight_decay" : 0.01,
            "epochs" : 50}

        for name, value in params.items():
            log_param(name, value)



        text_encoder = TextEncoder(DEVICE, model_name)
        visual_encoder = VisualEncoder(DEVICE, model_name)
        model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(DEVICE)
        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=params["learning_rate"],
                                      weight_decay=params["weight_decay"])

        schedule = get_constant_schedule_with_warmup(optimizer,
                                          num_warmup_steps=params["lr_warm_up_steps"])

        for i in range(params["epochs"]):
            accumulated_loss = 0
            dataloader = DataLoader(dataset, batch_size=params["batch_size"], collate_fn=collate_fn, shuffle=True)
            #fix the data loading issue
            for batch in tqdm(dataloader, desc=f"Epoch: {i}"):
                    
                batch_images, targets = collate_fn(text_model, static_visual_encoder, batch)

                outputs = model(**batch_images)

                pooled_outputs = outputs.image_embeds

                pooled_outputs_normalized = torch.nn.functional.normalize(pooled_outputs)
                targets_normalized = torch.nn.functional.normalize(targets)

                current_batch_size, _ = pooled_outputs.size()
                targets = torch.arange(0, current_batch_size).to(DEVICE)
                loss_a = ce_loss(pooled_outputs_normalized, targets)
                loss_b = ce_loss(targets_normalized, targets)
                loss = (loss_a + loss_b) / 2
                loss.backward()
                accumulated_loss += loss.item()

                if i % params["gradient_accumulation_steps"]:
                    optimizer.step()
                    optimizer.zero_grad()
                    schedule.step()
            epoch_loss =  accumulated_loss / len(dataloader)
            log_metric("loss", epoch_loss)
            print(epoch_loss)

        text_path, visual_path = model_repo.get_savepath(model_name_saved) 

        mlflow.pytorch.save_model(text_model, text_path)
        mlflow.pytorch.save_model(visual_model, visual_path)
