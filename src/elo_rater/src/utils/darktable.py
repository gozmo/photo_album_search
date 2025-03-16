import os
import glob
from xml.etree import ElementTree as ET

def _read_xmp_file(filepath, xmp_path):
    if "xmp" not in filepath:
        xmp_filepath = f"{filepath}.xmp"

    if not os.path.isfile(xmp_filepath):
        return []

    elements = []
    tree = ET.parse(xmp_filepath)
    root = tree.getroot()
    children = root.findall(PATH)

    if 0 < len(children):
        for child in children:
            elements.append(child.text)
    return elements

def read_tags(filepath):
    path = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/" + \
           "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description/" + \
           "{http://purl.org/dc/elements/1.1/}subject/" + \
           "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag/"

    return _read_xmp_file(filepath, path)

def read_stars(filepath):
    if "xmp" not in filepath:
        xmp_filepath = f"{filepath}.xmp"
    else:
        xmp_filepath = filepath
    tree = ET.parse(xmp_filepath)
    root = tree.getroot()
    
    path = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF/"
    elem =tree.findall(path)[0]
    rating = elem.get('{http://ns.adobe.com/xap/1.0/}Rating')
    return int(rating)

class Crawler:
    def __init__(self, root_directory, file_types):
        self.root_directory = root_directory
        self.file_types = file_types
        self.files = []

    def start(self):
        files = []
        for file_type in self.file_types:
            pattern = f"**/*.{file_type}"
            print(pattern)
            files += glob.glob(pattern,
                               recursive=True,
                               root_dir=self.root_directory)

        self.files = [f"{self.root_directory}{file}" for file in files]
        length = len(self.files)
        print(f"Crawler found {length} files")

        return self.files
