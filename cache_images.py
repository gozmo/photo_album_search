from crawler import Crawler
import pudb
import argparse
from image_cache import ImageCache
from image_cache import load_cached_image
from image_cache import get_cached_files
from io_utils import init_directory_structure


parser = argparse.ArgumentParser()
parser.add_argument("root_directory")

args = parser.parse_args()


crawler = Crawler(args.root_directory, ["jpg", "CR2","cr2"])

files = crawler.start()


init_directory_structure()
image_cache = ImageCache()
image_cache.cache(files)

