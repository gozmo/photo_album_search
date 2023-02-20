from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm


class VisualEncoder:
    def __init__(self, device):
        self.device = device
        # self.model = CLIPVisionModelWithProjection.from_pretrained("trained_clip_visual_model.pt").to(self.device)
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

    def encode(self, images):
        images = images.to(self.device)
        outputs = self.model(**images)

        pooled_outputs = outputs.image_embeds

        return pooled_outputs

class TextEncoder:
    def __init__(self, device):
        self.device = device
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_projection = self.text_model(**inputs).text_embeds

        return text_projection.to(self.device)
