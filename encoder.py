
from PIL import Image

from transformers import AutoProcessor
from transformers import CLIPVisionModel
from tqdm import tqdm


class Encoder:
    def __init__(self, files):
        self.files = files
        self.device = "cuda"
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.milvus_connection = Collection("image_vectors")      # Get an existing collection.

    def run(self):
        for filepath in tqdm(self.files):
            image = self.read_image(filepath)
            vector = self.encode(image)
            self.upload(filepath, vector)

    def read_image(self, filepath):
        with Image.open(filepath) as raw_image:

                
            image = self.processor(images=raw_image,
                                   return_tensors="pt",
                                   do_convert_rgb=True,
                                   do_resize=True
                                   )
            image = image.to(self.device)

        return image

    def encode(self, image):
        outputs = self.model(**image)

        # last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output[0].tolist()

        return pooled_output

    def upload(self, filepath, vector):
        response = self.milvus_connection.insert((vector, filepath))
