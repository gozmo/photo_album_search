from pprint import pprint
from transformers import pipeline
import requests
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from torch import nn
from transformers import AutoProcessor, CLIPModel
import torch
import pudb


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
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


inputs = processor( text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) 

labels_for_classification =  ["cat",
                              "remote",
                              "cat and remote",
                              "cat and dog", 
                              "lion and cheetah", 
                              "rabbit and lion"]
inputs = processor(images=image, return_tensors="pt")
vision_projection = vision_model(**inputs).image_embeds

text_similarities = []
for text in labels_for_classification:
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_projection = text_model(**inputs).text_embeds
    similarity = calculate_simlarity(vision_projection, text_projection)
    text_similarities.append(similarity)
text_similarities = torch.tensor(text_similarities)

probabilities = text_similarities.softmax(dim=0).tolist()
pprint(list(zip(labels_for_classification, probabilities)))

