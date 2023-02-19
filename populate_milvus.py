from crawler import Crawler
from encoder import VisualEncoder
import pudb
import argparse
from image_cache import ImageCache
from image_cache import load_cached_image
from image_cache import get_cached_files
from io_utils import init_directory_structure
from torch.utils.data import DataLoader 
from milvus import drop_collection
from milvus import connect
from milvus import upload_images
from milvus import create_collection
from transformers import AutoProcessor 

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def read_image(filepath):
    _ , extension = os.path.splitext(filepath)

    if extension.lower() == ".cr2":
        with rawpy.imread(filepath) as raw: 
            raw_image = raw.postprocess()
    else:
        raw_image = Image.open(filepath)

    images = processor(images=raw_image,
                       return_tensors="pt",
                       do_convert_rgb=True,
                       do_resize=True)
    return images


parser = argparse.ArgumentParser()
parser.add_argument("root_directory")
parser.add_argument("-d", "--delete", action="store_true")
args = parser.parse_args()

if args.delete:
    drop_collection()
    create_collection()

crawler = Crawler(args.root_directory, ["jpg", "CR2","cr2"])

files = crawler.start()

visual_encoder = VisualEncoder("cuda")


dataloader = DataLoader(files, batch_size=24)
for filepaths in tqdm(dataloader):
    encoded_images = []
    for filepath in filepaths:
        processed_image = read_image(filepath)
        encoded_image = visual_encoder.encode(processed_image)
        encoded_images.append(encoded_image)
    upload_image(filepaths, encoded_images)
