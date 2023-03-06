from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPVisionModelWithProjection
from transformers import AutoProcessor
from constants import DEFAULT_MODEL
import model_repo


class VisualEncoder:
    def __init__(self, device, model_name):
        self.device = device
        self.model = model_repo.get_visual_model(model_name).to(device)

    def encode(self, images):
        images = images.to(self.device)
        outputs = self.model(**images)

        pooled_outputs = outputs.image_embeds

        return pooled_outputs

class TextEncoder:
    def __init__(self, device, model_name):
        self.device = device
        self.clip_model = model_repo.get_text_model(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)

    def encode(self, text):
        text_input = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        text_projection = self.clip_model(**text_input).text_embeds

        return text_projection
