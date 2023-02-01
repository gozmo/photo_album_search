from transformers import pipeline
import requests
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel
from transformers import AutoProcessor, CLIPVisionModel
from torch import nn


text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels_for_classification =  ["cat",
                              "remote",
                              "cat and remote",
                              "cat and dog", 
                              "lion and cheetah", 
                              "rabbit and lion"]
text_embeddings = {}
for text in labels_for_classification:
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_embeddings[text] = text_model(**inputs).last_hidden_state

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image= Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = vision_model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output

import pudb
pu.db
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
for text, vector in text_embeddings.items():
    output = cos(vector, pooled_output)
    print(output, text)
