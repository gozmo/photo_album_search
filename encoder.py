from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm
from pymilvus import Collection 
import os
import rawpy
from torch.utils.data import DataLoader 


class Encoder:
    def __init__(self, files):
        self.files = files
        self.device = "cuda"
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.collection= Collection("image_vectors")      # Get an existing collection.

    def run(self):
        index_params = {
                  "metric_type":"L2",
                    "index_type":"IVF_FLAT",
                      "params":{"nlist":1024}
                      }

        self.collection.create_index(
                  field_name="vector", 
                    index_params=index_params
                    )

        files = DataLoader(self.files, batch_size=64)
        for filepath_batch in tqdm(files):
            images = self.read_image(filepath_batch)
            vectors = self.encode(images)
            self.upload(filepath_batch, vectors)


    def read_image(self, filepaths):
        raw_images = []
        for filepath in filepaths:
            _ , extension = os.path.splitext(filepath)

            if extension.lower() == ".cr2":
                with rawpy.imread(filepath) as raw:
                    raw_image = raw.postprocess()
            else:
                raw_image = Image.open(filepath)
            raw_images.append(raw_image)

        images = self.processor(images=raw_images,
                               return_tensors="pt",
                               do_convert_rgb=True,
                               do_resize=True
                               )
        images = images.to(self.device)
        return images

    def encode(self, images):
        outputs = self.model(**images)

        pooled_outputs = outputs.image_embeds.tolist()

        return pooled_outputs

    def upload(self, filepaths, vectors):
        ids = [hash(filepath) for filepath in filepaths]

        self.collection.insert([ids, vectors, filepaths])


class TextEncoder:
    def __init__(self):
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_projection = self.text_model(**inputs).text_embeds
        return text_projection

