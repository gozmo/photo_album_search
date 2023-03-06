import argparse
from constants import DEFAULT_MODEL
import pudb
import torch
import trainer

from encoder import VisualEncoder
from image_cache import get_cached_files
from image_cache import load_cached_image
from torch.utils.data import DataLoader 
from tqdm import tqdm
from transformers.image_processing_utils import BatchFeature
import db_utils
from constants import DEFAULT_MODEL
import mlflow

DEVICE="cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None)
args = parser.parse_args()


db_utils.connect()

model_name = ""
with mlflow.start_run() as run:
    if args.model_name== None:
        model_name = run.info.run_name
    else:
        model_name = args.model_name

    trainer.train(model_name)
mlflow.end_run()

db_utils.add_model_name(model_name)


collection_name = db_utils.model_name_to_collection_name(model_name)

db_utils.drop_collection(collection_name)
db_utils.create_collection(collection_name)

visual_encoder = VisualEncoder(DEVICE, model_name)


files = get_cached_files()

dataloader = DataLoader(files, batch_size=36)
for filepaths in tqdm(dataloader, desc="uploading"):
    cached_images = [load_cached_image(filepath) for filepath in filepaths]
    images = [elem[0][0] for elem in cached_images]
    filepaths = [elem[2] for elem in cached_images]
    data = {"pixel_values": torch.tensor(images)}
    batch = BatchFeature(data, tensor_type="pt").to(DEVICE)
    encoded_images = visual_encoder.encode(batch)

    db_utils.upload_images(collection_name, filepaths, encoded_images.tolist())
