from os.path import isfile
import xml.etree.ElementTree as ET
import glob
import os
import json
from collections import Counter
from tqdm import tqdm

RATINGS_CACHE = "cache/ratings.json"

def get_rating(filepath):
    
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return None
    
    root = tree.getroot()
    children = root.findall("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/")

    
    rating = None
    if 0 < len(children):

        for child in children:
            rating = child.get('{http://ns.adobe.com/xap/1.0/}Rating')
    return rating

def get_filename(filepath):
    filename = os.path.basename(filepath)
    name, extension = os.path.splitext(filename)
    return name


def get_all_ratings():

    ratings = {}
    
    pattern = "/media/yama/bilder/camera_photos/**/**/**/*.xmp"
    for filepath in tqdm(glob.glob(pattern,
                       recursive=True,
                       root_dir="/")):
        rating = get_rating(filepath)
        filename = get_filename(filepath)

        elem = {"filepath": filepath,
                "rating": rating,
                "filename": filename}
        ratings[filename] = elem

    return ratings

def get_all_ratings_cached():
    if os.path.isfile("cache/ratings.json"):
        with open("cache/ratings.json", "r") as f:
            ratings = json.load(f)

    else:
        ratings = get_all_ratings()
        with open("cache/ratings.json", "w") as f:
            json.dump(ratings, f)

    return ratings

if __name__ == "__main__":
    a = get_all_ratings_cached()
    from collections import Counter
    from pprint import pprint
    pprint(Counter(a.values()))

