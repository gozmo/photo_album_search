import os
import pickle
import rawpy
import glob
import pudb
import logging

from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor 
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np

from constants import Directories
from constants import DEFAULT_MODEL
from encoder import VisualEncoder

class ImageCache:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.visual_encoder = VisualEncoder("cuda", DEFAULT_MODEL)

    def cache(self, filepaths):

        data_loader = DataLoader(filepaths,
                                 collate_fn=self.collate_fn,
                                 batch_size=80, 
                                 num_workers=5)

        for filepaths, images, tags, ratings in tqdm(data_loader):
            embeddings = self.__embed(images)
            embeddings = embeddings.tolist()

            for i in range(len(ratings)):
                image_downsampled = images["pixel_values"][i]
                image_embedding = embeddings[i]
                image_tags = tags[i]
                image_rating = ratings[i]
                image_filepath = filepaths[i]

                # self.__write_image(image_filepath,
                                   # image_downsampled,
                                   # image_tags,
                                   # image_embedding,
                                   # image_rating)

    def collate_fn(self, filepaths):
        images = []
        tags = []
        ratings = []
        for filepath in filepaths:
            try:
                image, image_tags, image_rating = self.__read_file(filepath)
            except:
                logging.error(f"Failed to read file: {filepath}")
                continue

            images.append(image)
            tags.append(image_tags)
            ratings.append(image_rating)


        images = self.processor(images=images,
                                return_tensors="pt",
                                do_convert_rgb=True,
                                do_resize=True)

        return filepaths, images, tags, ratings

    def __read_file(self, filepath):

        try:
            tags, rating = self.__read_xmp(filepath)
        except:
            tags = None
            rating = None
           
        image = self.__read_image(filepath)
        return image, tags, rating


    def __read_image(self, filepath):
        _ , extension = os.path.splitext(filepath)

        if extension.lower() == ".cr2":
            with rawpy.imread(filepath) as raw: 
                raw_image = raw.postprocess()
        else:
            raw_image = Image.open(filepath)

        return raw_image

    def __embed(self, images):
        embeddings = self.visual_encoder.encode(images)
        return embeddings

    def __read_xmp(self, filepath):
        tags = []
        rating = None

        xmp_filepath = f"{filepath}.xmp"
        if not os.path.isfile(xmp_filepath):
            return tags, rating


        tree = ET.parse(xmp_filepath)
        root = tree.getroot()

        try:
            tags = self.__get_xmp_tags(root)
        except:
            pass

        try:
            rating = self.__get_xmp_rating(root)
        except:
            pass

        return tags, rating

    def __get_xmp_tags(self, root):
        PATH = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/" + \
               "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description/" + \
               "{http://purl.org/dc/elements/1.1/}subject/" + \
               "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag/"
        children = root.findall(PATH)

        tags = []
        if 0 < len(children):
            for child in children:
                tags.append(child.text)
        return tags

    def __get_xmp_rating(self, root):
        children = root.findall("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/")

        rating = None
        if 0 < len(children):

            for child in children:
                rating = child.get('{http://ns.adobe.com/xap/1.0/}Rating')
        return rating

    def __get_name(self, original_filepath):
        filename_and_ext = os.path.basename(original_filepath)
        filename, _ = os.path.splitext(filename_and_ext)
        filepath = f"{Directories.IMAGE_CACHE}/{filename}.cache"
        if os.path.isfile(filepath):
            filepath = f"{Directories.IMAGE_CACHE}/{filename}_2.cache"

        return filepath

    def __write_image(self, filepath, processed_images, tags, embedding, rating):
        content = {"image": processed_images["pixel_values"].tolist(),
                   "tags": tags,
                   "rating": rating,
                   "embedding": embedding,
                   "filepath": filepath}

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

    return content

