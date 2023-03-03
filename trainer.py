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
import logging
import constants
import model_repo
from mlflow import log_metric, log_param, log_artifacts
import mlflow


DEVICE = "cuda"

def generate_sentence(tags):
    if len(tags) == 0:
        return "a picture"

    sentence = "a picture of "
    tag_subsentence = "and".join(tags)

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

    for image, image_tags in zip(batch_images["pixel_values"], tags):
        if eval_tag in image_tags:
            image = image.detach().to("cpu")
            output.append((image, eval_tag))

    return output


def eval_distance(visual_model, text_encoder, eval_set):

    distances = 0
    for image, tags in eval_set:
        image = image.resize(1,3,224,224).to(DEVICE)
        outputs = visual_model(pixel_values=image)
        image_embedding = outputs.image_embeds[0]

        sentence = generate_sentence(tags)
        text_embedding = text_encoder.encode(sentence)

        image_embedding_normalized = torch.nn.functional.normalize(image_embedding,dim=0)
        text_embedding_normalized = torch.nn.functional.normalize(text_embedding, dim=0)

        distance = (image_embedding_normalized - text_embedding_normalized).norm()

        distances += distance.item()

    avg_distance = distances / len(eval_set)
    return avg_distance

def train(model_name_saved):
    with mlflow.start_run():
        model_name = constants.DEFAULT_MODEL
        dataset = []

        logging.info("Image files will be read on demand in collate_fn")
        dataset = get_cached_files()
        
        # dataset = dataset[:500]

        params = {
            "batch_size" : 36,
            "lr_warm_up_steps" : 5,
            "gradient_accumulation_steps" : 100,
            "learning_rate" : 1e-5,
            "weight_decay" : 0.01,
            "epochs" : 10,
            "eval_set_size": 20}

        for name, value in params.items():
            log_param(name, value)


        clip_model = CLIPModel.from_pretrained(constants.DEFAULT_MODEL).to(DEVICE)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        ce_loss_a = torch.nn.CrossEntropyLoss()
        ce_loss_b = torch.nn.CrossEntropyLoss()
        optimizer= torch.optim.AdamW(clip_model.parameters(),
                                              lr=params["learning_rate"],
                                              weight_decay=params["weight_decay"])

        schedule= get_constant_schedule_with_warmup(optimizer,
                                          num_warmup_steps=params["lr_warm_up_steps"])

        eval_set = []

        step_count = 0
        for i in range(params["epochs"]):
            accumulated_loss = 0
            step_loss = 0
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=params["batch_size"], shuffle=True)
            

            for images, tags in tqdm(dataloader, desc=f"Epoch: {i}"):
                    
                batch = create_training_data(processor, images, tags)


                # if len(eval_set) < params["eval_set_size"]:
                    # eval_set += find_eval_images(batch_images, tags, "algot")

                outputs = clip_model(**batch)
                
                image_embeddings = outputs.image_embeds
                text_embeddings = outputs.text_embeds

                # image_embeddings_normalized = image_embeddings
                # text_embeddings_normalized = text_embeddings

                image_embeddings_normalized = torch.nn.functional.normalize(image_embeddings)
                image_embeddings_normalized.retain_grad()

                text_embeddings_normalized= torch.nn.functional.normalize(text_embeddings)
                text_embeddings_normalized.retain_grad()

                current_batch_size, _ = image_embeddings.size()
                contrastive_targets = torch.arange(0, current_batch_size).to(DEVICE)

                loss_a = ce_loss_a(image_embeddings_normalized, contrastive_targets)
                loss_b = ce_loss_b(text_embeddings_normalized, contrastive_targets)

                loss = (loss_a + loss_b) / 2
                loss.backward()

                accumulated_loss += loss.item()
                step_loss += loss.item()

                if step_count % params["gradient_accumulation_steps"] == 0:
                    pu.db
                    optimizer.step()
                    optimizer.zero_grad()
                    schedule.step()

                    log_metric("step loss", step_loss, step=step_count)
                    step_loss = 0

                    # avg_distance = eval_distance(visual_model, text_encoder, eval_set)
                    # log_metric("avg distance", avg_distance, step=step_count)
                step_count += 1

            epoch_loss =  accumulated_loss / len(dataloader)
            log_metric("epoch loss", epoch_loss, step=i)


        text_path, visual_path = model_repo.get_savepath(model_name_saved) 

        text_model.save_pretrained(text_path)
        visual_model.save_pretrained(visual_path)
