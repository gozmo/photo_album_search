from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm
from pymilvus import Collection 


class Encoder:
    def __init__(self, files):
        self.files = files
        self.device = "cuda"
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.collection= Collection("image_vectors")      # Get an existing collection.

    def run(self):
        for filepath in tqdm(self.files):
            image = self.read_image(filepath)
            vector = self.encode(image)
            self.upload(filepath, vector)

        index_params = {
                  "metric_type":"L2",
                    "index_type":"IVF_FLAT",
                      "params":{"nlist":1024}
                      }

        self.collection.create_index(
                  field_name="vector", 
                    index_params=index_params
                    )

    def read_image(self, filepath):
        basepath, extension = os.path.splitext(filepath)

        if extension.lower() == ".cr2":
            with rawpy.imread(path) as raw:
                raw_image = raw.postprocess()
        else:
            raw_image = Image.open(filepath)

        image = self.processor(images=raw_image,
                               return_tensors="pt",
                               do_convert_rgb=True,
                               do_resize=True
                               )
        image = image.to(self.device)

        return image

    def encode(self, image):
        outputs = self.model(**image)

        pooled_output = outputs.image_embeds[0].tolist()

        return pooled_output

    def upload(self, filepath, vector):
        id = hash(filepath)

        response = self.collection.insert([[id], [vector], [filepath]])


class TextEncoder:
    def __init__(self):
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_projection = self.text_model(**inputs).text_embeds
        return text_projection

