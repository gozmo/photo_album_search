
from PIL import Image
# import requests

from transformers import AutoProcessor
from transformers import CLIPVisionModel
from tqdm import tqdm

# device = "cuda"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True).to(device)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) 

class Encoder:
    def __init__(self, files):
        self.files = files
        self.device = "cuda"
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def run(self):
        for filepath in tqdm(self.files):
            image = self.read_image(filepath)
            self.encode(image)

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

        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
