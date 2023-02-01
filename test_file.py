from transformers import pipeline
import requests
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from torch import nn
from transformers import AutoProcessor, CLIPModel
import torch


def calculate_simlarity(image_embeds, text_embeds):

    # normalized features
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale_init_value = 2.6592
    logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
    logit_scale = logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.t()
    return logits_per_text


text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
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
    text_embeddings[text] = text_model(**inputs).text_embeds

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image= Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
vision_projection = vision_model(**inputs).image_embeds

import pudb
pu.db
for text, text_projection in text_embeddings.items():
    similarity = calculate_simlarity(vision_projection, text_projection)
    print(similarity, text)
