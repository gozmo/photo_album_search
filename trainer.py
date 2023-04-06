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
import optuna
import mlflow
import random

DEVICE = "cuda"

def generate_sentence(tags):
    if len(tags) == 0:
        return "a photo"

    sentence = "a photo of "
    tag_subsentence = " and ".join(tags)

    return sentence + tag_subsentence

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def collate_fn(self, filepaths):
        images = []
        tags = []
        embeddings = []

        for filepath in filepaths:
            image, image_tags, _, embedding = load_cached_image(filepath)
            images.append(image[0])
            tags.append(image_tags)
            embeddings.append(embedding)

        sentences = [generate_sentence(image_tags) for image_tags in tags]
        text_input = self.processor(text=sentences, padding=True)

        data = {"pixel_values": torch.tensor(images)}
        data.update(text_input)
        batch = BatchFeature(data, tensor_type="pt").to(DEVICE)

        return batch, embeddings, tags

# def create_training_data(text_model,tokenizer, static_visual_encoder, images, tags):

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

def __train_loop(params, trial=None, experiment_id=None):
    model_name_saved = None

    with mlflow.start_run(experiment_id=experiment_id) as run:
        
        model_name_saved = run.info.run_name


        dataset = []

        logging.info("Image files will be read on demand in collate_fn")
        dataset = get_cached_files()

        random.shuffle(dataset)

        idx = int(len(dataset) * params["data_size"])
        dataset = dataset[:idx]
        


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

        collator = Collator(processor)

        global_step = 0
        for epoch_i in range(params["epochs"]):
            accumulated_loss = 0
            step_loss = 0
            dataloader = DataLoader(dataset, collate_fn=collator.collate_fn, batch_size=params["batch_size"], shuffle=True)
            

            for batch, orig_visual_embeddings, tags in tqdm(dataloader, desc=f"Epoch: {i}"):

                outputs = clip_model(**batch)

                image_embeddings = outputs.image_embeds
                text_embeddings = outputs.text_embeds

                for i, image_tags in enumerate(tags):
                    if len(image_tags) == 0:
                        
                        original_embedding = torch.tensor(orig_visual_embeddings[i]).to(DEVICE)
                        text_embeddings[i] = original_embedding

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

                if global_step % params["gradient_accumulation_steps"] == 0 and global_step != 0:
                    
                    all_grads = [ layer.grad.norm().item() for layer in clip_model.parameters()]
                    all_grads = sum(all_grads)
                    log_metric("all_grads", all_grads, step=global_step)

                    all_weights = [ layer.norm().item() for layer in clip_model.parameters()]
                    all_weights = sum(all_weights)
                    log_metric("all_weights ", all_weights, step=global_step)


                    optimizer.step()
                    optimizer.zero_grad()
                    schedule.step()

                    step_loss /= params["gradient_accumulation_steps"]
                    log_metric("step loss", step_loss, step=global_step)
                    step_loss = 0

                    if trial:
                        trial.report(step_loss, global_step)

                        if trial.should_prune():
                            raise optuna.TrialPruned()

                global_step += 1

            epoch_loss =  accumulated_loss / len(dataloader)
            log_metric("epoch loss", epoch_loss, step=epoch_i)


        save_path = model_repo.get_savepath(model_name_saved, "clip") 

        clip_model.save_pretrained(save_path)
        return model_name_saved, epoch_loss

def train():

    params = {
        "batch_size" : 60,
        "gradient_accumulation_steps" : 2,
        "learning_rate" : 1e-5,
        "weight_decay" : 0.01,
        "epochs" : 2,
        "data_size": 1.0,
        "learning_rate_scheduler": "cosine"}

    model_name, _ = __train_loop(params)
    return model_name 

def __hpo_objective(trial, experiment_id, data_size):

    params = {
        "batch_size" : 60,
        "gradient_accumulation_steps" : trial.suggest_int("gradient_accumulation_steps",1,20),
        "learning_rate" : trial.suggest_categorical("learning_rate", [1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8]),
        "weight_decay" : trial.suggest_categorical("weight_decay", [0.01,0.05, 0.1, 0.2, 0.3, 0.5]),
        "epochs" : 10,
        "data_size": data_size,
        "learning_rate_scheduler": trial.suggest("lr_schedule", ["constant", "cosine"])
        }
    model_name_saved, loss = __train_loop(params, trial, experiment_id)

    trial.set_user_attr("model_name", model_name_saved)
    return loss


def run_hpo(experiment_name):
    experiment_id = mlflow.create_experiment(f"hpo run {experiment_name}")

    study = optuna.create_study(direction='minimize',
                                # pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                      # max_resource=10,
                                                                      # reduction_factor=3)
                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_early_stopping_rate=1,
                                                                              bootstrap_count=1,
                                                                              min_resource=1)
                                )

    objective = lambda trial: __hpo_objective(trial, experiment_id, data_size=0.05)
    study.optimize(objective, n_trials=30)
    trials = [(t.value, t.user_attrs["model_name"]) for t in study.get_trials()]
    (value, model_name) = max(trials)

    return model_name
