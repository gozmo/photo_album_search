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

DEVICE="cuda"

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
args = parser.parse_args()

db_utils.add_model_name(args.model_name)

db_utils.connect()

trainer.train(args.model_name)

# if args.model_name == "default":
    # model_name = DEFAULT_MODEL
# else:
    # model_name = args.model_name


# collection_name = db_utils.model_name_to_collection_name(args.model_name)

# db_utils.drop_collection(collection_name)
# db_utils.create_collection(collection_name)

# visual_encoder = VisualEncoder(DEVICE, model_name)


# files = get_cached_files()

# dataloader = DataLoader(files, batch_size=24)
# for filepaths in tqdm(dataloader, desc="uploading"):
    # cached_images = [load_cached_image(filepath) for filepath in filepaths]
    # images = [elem[0][0] for elem in cached_images]
    # filepaths = [elem[2] for elem in cached_images]
    # data = {"pixel_values": torch.tensor(images)}
    # batch = BatchFeature(data, tensor_type="pt").to(DEVICE)
    # encoded_images = visual_encoder.encode(batch)

    # db_utils.upload_images(collection_name, filepaths, encoded_images.tolist())
