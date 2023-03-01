import os
import pickle
import rawpy
import glob
import random

from PIL import Image
from torch.utils.data import DataLoader 
from torch.utils.data import DataLoader 
from transformers import AutoProcessor 
import xml.etree.ElementTree as ET
from tqdm import tqdm

from constants import Directories

class ImageCache:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def cache(self, filepaths):

        for filepath in tqdm(filepaths):
            try:
                tags = self.__read_xmp(filepath)
                processed_image = self.__read_image(filepath)
                self.__cache_image(filepath, processed_image, tags)
            except:
                pass


    def __read_image(self, filepath):
        _ , extension = os.path.splitext(filepath)

        if extension.lower() == ".cr2":
            with rawpy.imread(filepath) as raw: 
                raw_image = raw.postprocess()
        else:
            raw_image = Image.open(filepath)

        images = self.processor(images=raw_image,
                                return_tensors="pt",
                                do_convert_rgb=True,
                                do_resize=True)
        return images

    def __read_xmp(self, filepath):
        xmp_filepath = f"{filepath}.xmp"
        if not os.path.isfile(xmp_filepath):
            return []

        PATH = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/" + \
               "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description/" + \
               "{http://purl.org/dc/elements/1.1/}subject/" + \
               "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag/"

        tags = []
        tree = ET.parse(xmp_filepath)
        root = tree.getroot()
        children = root.findall(PATH)

        if 0 < len(children):
            for child in children:
                tags.append(child.text)
        return tags

    def __get_name(self, original_filepath):
        filename_and_ext = os.path.basename(original_filepath)
        filename, _ = os.path.splitext(filename_and_ext)
        filepath = f"{Directories.IMAGE_CACHE}/{filename}.cache"
        return filepath

    def __cache_image(self, filepath, processed_images, tags):
        content = {"images": processed_images["pixel_values"].tolist(),
                   "tags": tags,
                   "filepaths": filepath}

        filename = self.__get_name(filepath)
        with open(filename, "wb") as f:
            pickle.dump(content, f) 

def get_cached_files():
    cache_files = glob.glob(f"{Directories.IMAGE_CACHE}/*.cache",
                            recursive=False,
                            root_dir=".")
    return cache_files

def load_cached_image(cache_file):
    with open(cache_file, "rb") as f:
        content = pickle.load(f)
    image = content["images"]
    # tags = content["tags"]
    tags = random.choice([["algot"], ["majken"], [] ])
    filepath = content["filepaths"]

    return image, tags, filepath

