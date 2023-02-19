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


# setup_milvus()
crawler = Crawler(args.root_directory, ["jpg", "CR2","cr2"])

files = crawler.start()

# encoder = Encoder(files)
# encoder.run()

init_directory_structure()
image_cache = ImageCache()
image_cache.cache(files)


files = get_cached_files()
for file in files:
    load_cached_image(file)
