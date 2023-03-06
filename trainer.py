import pudb

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader 
from image_cache import load_cached_image
from image_cache import get_cached_files
from transformers.image_processing_utils import BatchFeature
from transformers import CLIPModel
from transformers import AutoProcessor
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import logging
import constants
import model_repo
from mlflow import log_metric, log_param 

DEVICE = "cuda"

def generate_sentence(tags):
    if len(tags) == 0:
        return "a picture"

    sentence = "a picture of "
    tag_subsentence = " and ".join(tags)

    return sentence + tag_subsentence

def collate_fn(filepaths):
    images = []
    tags = []

    for filepath in filepaths:
        image, image_tags, _ = load_cached_image(filepath)
        images.append(image[0])
        tags.append(image_tags)
    return images, tags

def create_training_data(processor, images, tags):
    sentences = [generate_sentence(image_tags) for image_tags in tags]
    text_input = processor(text=sentences, padding=True)

    data = {"pixel_values": torch.tensor(images)}
    data.update(text_input)
    batch = BatchFeature(data, tensor_type="pt").to(DEVICE)

    return batch  

# def create_training_data(text_model,tokenizer, static_visual_encoder, images, tags):

    # text_embeddings = torch.zeros(len(images), 512, requires_grad=True).to(DEVICE)
    # text_embeddings.retain_grad()
    # for i, image_tags in enumerate(tags):
        # if 0 < len(image_tags):
            # sentence = generate_sentence(image_tags)
            # inputs = tokenizer(sentence, padding=True, return_tensors="pt").to(DEVICE)
            # target = text_model(**inputs).text_embeds
            # target.retain_grad()
        # else:
            # data = {"pixel_values": torch.tensor([ images[i] ])}
            # input_image = BatchFeature(data, tensor_type="pt")
            # target = static_visual_encoder.encode(input_image).detach()
            # target.requires_grad = True

        # text_embeddings[i] = torch.clone(target[0])
        
    # data = {"pixel_values": torch.tensor(images)}
    # images = BatchFeature(data, tensor_type="pt").to(DEVICE)

    # return images, text_embeddings 


def find_eval_images(batch_images, tags, eval_tag):
    output = []

    for image, image_tags in zip(batch_images, tags):
        if eval_tag in image_tags:
            image = torch.tensor(image)
            output.append((image, eval_tag))

    return output


def eval_distance(clip_model, processor, eval_set):

    distances = 0
    for image, tags in eval_set:
        image = image.resize(1,3,224,224).to(DEVICE)

        sentences = generate_sentence(tags)
        text_input = processor(text=sentences, padding=True, return_tensors="pt").to(DEVICE)
        data = {"pixel_values": image}
        data.update(text_input)

        output = clip_model(**data).detach()
        image_embedding = output.image_embeds
        text_embedding = output.text_embeds

        image_embedding_normalized = torch.nn.functional.normalize(image_embedding,dim=0)
        text_embedding_normalized = torch.nn.functional.normalize(text_embedding, dim=0)

        distance = (image_embedding_normalized - text_embedding_normalized).norm()

        distances += distance.item()

    avg_distance = distances / len(eval_set)
    return avg_distance

def train(model_name_saved):
        dataset = []

        logging.info("Image files will be read on demand in collate_fn")
        dataset = get_cached_files()
        

        params = {
            "batch_size" : 36,
            "gradient_accumulation_steps" : 20,
            "learning_rate" : 1e-5,
            "weight_decay" : 0.01,
            "epochs" : 10,
            "eval_set_size": 20,
            "learning_rate_scheduler": "cosine"}

        for name, value in params.items():
            log_param(name, value)


        clip_model = CLIPModel.from_pretrained(constants.DEFAULT_MODEL).to(DEVICE)
        clip_model.requires_grad_ = True

        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        ce_loss_a = torch.nn.CrossEntropyLoss()
        ce_loss_b = torch.nn.CrossEntropyLoss()
        optimizer= torch.optim.AdamW(clip_model.parameters(),
                                      lr=params["learning_rate"],
                                      weight_decay=params["weight_decay"])

        total_steps = int((len(dataset) / params["batch_size"]) * params["epochs"])
        warmup = int(0.2 * params["epochs"])
        if params["learning_rate_scheduler"] == "cosine":
            schedule= get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                         num_warmup_steps=warmup,
                                                                         num_cycles=10,
                                                                         num_training_steps=total_steps)
        elif params["learning_rate_scheduler"] == "constant":
            schedule= get_constant_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup)

        eval_set = []

        step_count = 0
        for i in range(params["epochs"]):
            accumulated_loss = 0
            step_loss = 0
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=params["batch_size"], shuffle=True)
            

            for images, tags in tqdm(dataloader, desc=f"Epoch: {i}"):
                    
                batch = create_training_data(processor, images, tags)


                # if len(eval_set) < params["eval_set_size"]:
                    # eval_set += find_eval_images(images, tags, "algot")

                outputs = clip_model(**batch)
                
                image_embeddings = outputs.image_embeds
                text_embeddings = outputs.text_embeds

                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp()

                logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
                logits_per_text = logit_scale * text_embeddings @ image_embeddings.t()


                current_batch_size, _ = image_embeddings.size()
                contrastive_targets = torch.arange(0, current_batch_size).to(DEVICE)

                loss_a = ce_loss_a(logits_per_image, contrastive_targets)
                loss_b = ce_loss_b(logits_per_text, contrastive_targets)

                loss = (loss_a + loss_b) / 2
                loss.backward()

                accumulated_loss += loss.item()
                step_loss += loss.item()

                if step_count % params["gradient_accumulation_steps"] == 0 and step_count != 0:
                    
                    all_grads = [ layer.grad.norm().item() for layer in clip_model.parameters()]
                    all_grads = sum(all_grads)
                    log_metric("all_grads", all_grads, step=step_count)

                    all_weights = [ layer.norm().item() for layer in clip_model.parameters()]
                    all_weights = sum(all_weights)
                    log_metric("all_weights ", all_weights, step=step_count)


                    optimizer.step()
                    optimizer.zero_grad()
                    schedule.step()

                    step_loss /= params["gradient_accumulation_steps"]
                    log_metric("step loss", step_loss, step=step_count)
                    step_loss = 0

                    # avg_distance = eval_distance(clip_model, processor, eval_set)
                    # log_metric("avg distance", avg_distance, step=step_count)

                step_count += 1

            epoch_loss =  accumulated_loss / len(dataloader)
            log_metric("epoch loss", epoch_loss, step=i)


        save_path = model_repo.get_savepath(model_name_saved, "clip") 

        clip_model.save_pretrained(save_path)
