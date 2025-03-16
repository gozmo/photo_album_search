from torch.utils.data.sampler import WeightedRandomSampler
from src.star_prediction.xml_reader import get_all_ratings_cached
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import nn
import rawpy
from PIL import Image
import os
from transformers import AutoImageProcessor, get_constant_schedule_with_warmup
from tqdm import tqdm
import mlflow
from sklearn.metrics import f1_score


from src.star_prediction.model import Model


import logging
logger = logging.getLogger(__name__)

#TODO: Move constants to new file
IMG_CACHE = "cache/images"

class RatingDataset(Dataset):
    def __init__(self, ratings, classify=False):
        if classify:
            self.ratings = [e for e in ratings if e['rating'] is None]
        else:
            self.ratings = [e for e in ratings if e['rating'] is not None]

    def __getitem__(self, index):
        return self.ratings[index]

    def __len__(self):
        return len(self.ratings)


#TODO: Move image IO to separate module, that could be used through out the project
def make_cached_filepath(filepath):
    basename= os.path.basename(filepath)
    filename, _ = os.path.splitext(basename)
    cached_filepath = f"{IMG_CACHE}/{filename}.jpg"
    return cached_filepath

def is_image_cached(path):
    cached_filepath = make_cached_filepath(path)
    return os.path.isfile(cached_filepath)


def __read_image(filepath):
    _ , extension = os.path.splitext(filepath)
    if extension.lower() == ".cr2":
        with rawpy.imread(filepath) as raw: 
            raw_image = raw.postprocess()
        raw_image = Image.fromarray(raw_image)
    else:
        raw_image = Image.open(filepath)
    return raw_image

def downsample_image(image):
    img = image.resize((320, 240))
    return img

def cache_image(filepath, image):
    
    cached_filepath = make_cached_filepath(filepath)
    image.save(cached_filepath)

def read_cached_image(path):
    cached_filepath = make_cached_filepath(path)
    return Image.open(cached_filepath)

def read_image(path):

    if not is_image_cached(path):
        original_image = __read_image(path)
        downsampled_image = downsample_image(original_image)
        cache_image(path, downsampled_image)

    return read_cached_image(path)

RATING_MAPPING = {"0": 0,
                  "1": 1,
                  "2": 2,
                  "3": 3,
                  "4": 4,
                  "5": 5,
                  "-1": 6}

SAMPLING_WEIGHT = {"0": 1,
                  "1": 1,
                  "2": 2,
                  "3": 5,
                  "4": 10,
                  "5": 20,
                  "-1": 10}


def rating_to_target(rating):
    target_tensor = torch.zeros(len(RATING_MAPPING))
    idx = RATING_MAPPING[rating]
    target_tensor[idx] = 1.0
    
    return target_tensor
    


#TODO: Collator for caching only
class Collator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def collate_fn(self, batch):
        images = []
        targets = []

        for elem in batch:
            try:
                rest, _ = os.path.splitext(elem['filepath'])
                image = read_image(rest)
                images.append(image)
            except:
                continue

            target = rating_to_target(elem['rating'])
            targets.append(target)


        img_tensors = processor(images, return_tensors="pt")

        return img_tensors.to(self.device), torch.stack(targets).to(self.device)

def training_step(dataloader,
                  model,
                  loss_fn,
                  optimizer_resnet,
                  optimizer_dense,
                  resnet_lr_schedule,
                  global_step):
    training_loss = 0
    targets_f1 = []
    predictions_f1 = []
    for batch in tqdm(dataloader, desc="training"):
        images, targets = batch

        probabilities = model(images)

        loss = loss_fn(probabilities, targets)

        optimizer_resnet.zero_grad()
        optimizer_dense.zero_grad()

        loss.backward()

        optimizer_resnet.step()
        optimizer_dense.step()
        resnet_lr_schedule.step()

        training_loss += loss.item()
        global_step += 1

        target = targets.argmax(dim=1).tolist()
        predictions = probabilities.argmax(dim=1).tolist()

        predictions_f1.extend(predictions)
        targets_f1.extend(target)

    mlflow.log_metric("training_loss",  training_loss, step=global_step)

    macro_f1 = f1_score(targets_f1, predictions_f1, average='macro', labels=list(RATING_MAPPING.keys()), zero_division=0.0)
    mlflow.log_metric(f"train_f1_macro",macro_f1, step=global_step)

    multiclass_f1_score = f1_score(targets_f1, predictions_f1, average=None, labels=list(RATING_MAPPING.keys()), zero_division=0.0)
    for class_name, idx in RATING_MAPPING.items():
        score = multiclass_f1_score[idx]
        mlflow.log_metric(f"train_f1_{class_name}", score, step=global_step)


def validation_step(validation_dataset,
                    model,
                    loss_fn,
                    global_step):

    validation_loss = 0
    i = 0


    targets_f1 = []
    predictions_f1 = []
    for batch in tqdm(validation_dataset, desc="validation"):
        images, targets = batch

        probabilities = model(images)

        loss = loss_fn(probabilities, targets)
        validation_loss += loss.item()

        target = targets.argmax(dim=1).tolist()
        predictions = probabilities.argmax(dim=1).tolist()

        predictions_f1.extend(predictions)
        targets_f1.extend(target)
        i += 1


    macro_f1 = f1_score(targets_f1, predictions_f1, average='macro', labels=list(RATING_MAPPING.keys()), zero_division=0.0)
    mlflow.log_metric(f"val_f1_macro_f1",macro_f1, step=global_step)

    multiclass_f1_score = f1_score(targets_f1, predictions_f1, average=None, labels=list(RATING_MAPPING.keys()), zero_division=0.0)
    for class_name, idx in RATING_MAPPING.items():
        score = multiclass_f1_score[idx]
        mlflow.log_metric(f"val_f1_{class_name}", score, step=global_step)

    mlflow.log_metric("validation_loss", validation_loss, step=global_step)

    return loss


if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name")
    parser.add_argument('--steps', type=int, default=0, help='num steps/batches per epoch')
    parser.add_argument('--epochs', type=int, default=2, help='num of epochs')
    parser.add_argument('--limit', type=int, default=-1, help='limit dataset size')
    parser.add_argument("--train", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--classify", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    batch_size = 20

    if args.cache:

        ratings_dict = get_all_ratings_cached()

        dataset = RatingDataset(ratings_dict.values())
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        collator = Collator(processor, device='cpu')
        dataloader = DataLoader(dataset,
                                batch_size=20,
                                shuffle=False,
                                num_workers=20,
                                collate_fn=collator.collate_fn)
        for batch in tqdm(dataloader):
            pass

    if args.classify:
        ratings_dict = get_all_ratings_cached()
        classify_dataset = RatingDataset(ratings_dict.values())
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        collator = Collator(processor, device='cuda')
        dataloader_classify = DataLoader(classify_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collator.collate_fn)
        # mlflow_run_id = "df444ca2766243a3967f407445c2d338"
        # run_relative_path_to_model = "resnet/data/"
        # model_path = f"runs:/{mlflow_run_id}/{run_relative_path_to_model}"
        full_path = "/home/goz/projects/photo_album_search/mlruns/0/df444ca2766243a3967f407445c2d338/artifacts/resnet/data/model.pth"
        model = torch.load(full_path, weights_only=False)

        from collections import Counter
        c = Counter()
        i = 0
        for batch in tqdm(dataloader_classify, desc="classifying"):
            images, targets = batch

            probabilities = model(images)

            predictions = probabilities.argmax(dim=1)

            c.update(predictions.tolist())
            i += 1

            if i > 10:
                break

        print(c)


    if args.train:

        mlflow.set_tracking_uri("http://localhost:5000")

        with mlflow.start_run():
            mlflow.set_experiment(args.experiment_name)
            mlflow.set_experiment_tag("debug", "debug")

            ratings_dict = get_all_ratings_cached()
            ratings_values = list(ratings_dict.values())

            if args.limit != -1:
                ratings_values = ratings_values[:args.limit]

            train_gen, val_gen = random_split(ratings_values, [0.98, 0.02])

            training_dataset = RatingDataset(list(train_gen))

            validation_dataset = RatingDataset(list(val_gen))

            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

            collator = Collator(processor, device='cuda')

            sample_weights = [SAMPLING_WEIGHT[e['rating']] for e in training_dataset]

            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(training_dataset))


            
            dataloader_train = DataLoader(training_dataset,
                                    batch_size=batch_size,
                                    sampler=sampler,
                                    collate_fn=collator.collate_fn)

            dataloader_validation = DataLoader(training_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=collator.collate_fn)

            epochs = args.epochs

            model = Model('cuda', RATING_MAPPING)

            optimizer_resnet = torch.optim.AdamW(model.resnet.parameters())
            resnet_lr_schedule = get_constant_schedule_with_warmup(optimizer=optimizer_resnet,
                                                                   num_warmup_steps=100)

            optimizer_dense = torch.optim.AdamW(model.dense.parameters())
            loss_fn = nn.BCELoss()

            best_loss = 600_000
            global_step = 0

            for epoch_i in range(epochs):
                training_step(dataloader_train,
                              model,
                              loss_fn,
                              optimizer_resnet,
                              optimizer_dense,
                              resnet_lr_schedule,
                              global_step)
                validation_loss = validation_step(dataloader_validation,
                                           model,
                                           loss_fn,
                                           global_step)

                if validation_loss < best_loss:
                    logger.info(f"Saving new model, new loss: {validation_loss}")
                    best_loss = validation_loss
                    model_info = mlflow.pytorch.log_model(pytorch_model=model,
                                                          artifact_path="resnet")





