import numpy as np
import gradio as gr
import glob
from PIL import Image
from PIL import ImageOps
import pudb
import random
from pprint import pprint
from collections import defaultdict
from src.utils.darktable import read_stars
import os
import csv
import tqdm
import logging

HEIGHT = 800
IMAGE_CACHE = "cache/images"

class Files:
    MATCH_HISTORY = "data/match_history.csv"

class ELOConstants:
    PLAYER1 = "player1"
    PLAYER2 = "player2"
    OUTCOME = "outcome"
    DRAW = "draw"
    FIELDNAMES = [PLAYER1, PLAYER2, OUTCOME]

class ImageLoader:
    def __init__(self):
        self.files = glob.glob("**/*.CR2",
                               root_dir=".")
        self.lhs_id = None
        recursive=True,
        self.rhs_id = None

        os.makedirs(IMAGE_CACHE, exist_ok=True)

    def __make_cached_image_filepath(self, filepath):
        basename = os.path.basename(filepath)
        filename, ext = os.path.splitext(basename)

        cached_image_filepath = f"{IMAGE_CACHE}/{filename}.jpg"
        
        return cached_image_filepath

    def __cache_image(self, filepath):
        image = Image.open(filepath)
        image = ImageOps.exif_transpose(image)

        width, height = image.size

        resized_image = image.resize( (width//2, height//2))


        filepath = self.__make_cached_image_filepath(filepath)
        with open(filepath, "w") as f:
            resized_image.save(f)

    def __is_cached(self, filepath):
        filepath = self.__make_cached_image_filepath(filepath)
        return os.path.isfile(filepath)


    def __read_cached_image(self, filepath):
        filepath = self.__make_cached_image_filepath(filepath)
        return Image.open(filepath)

    def __get_image(self, files_to_ignore=[]):

        random_space = set(self.files) - set(files_to_ignore)
        random_space = list(random_space)
       
        image_path = random.choice(random_space)
        image = self.read_image(image_path)

        return image_path, image

    def read_image(self, image_path):
        if not self.__is_cached(image_path):
            self.__cache_image(image_path)

        image = self.__read_cached_image(image_path)
        return image

    def lhs_fixed(self):
        if self.lhs_id:
            image = self.lhs_id
        else:
            image = self.lhs()
        return image

    def lhs(self):
        image_path, image = self.__get_image(files_to_ignore=[self.lhs_id])
        self.lhs_id = image_path
        return image

    def rhs(self):
        image_path, image = self.__get_image(files_to_ignore=[self.rhs_id])
        self.rhs_id = image_path
        return image
    
    def all_images(self):
        images = [self.read_image(path) for path in self.files]
        return images

class ELO:
    def __init__(self, default_rating=400):
        self.default_rating = default_rating
        self.ratings = {}
        self.__load_history()

    def __load_history(self):
        if not os.path.isfile(Files.MATCH_HISTORY):
            self.__create_match_history_file()
        
        f = open(Files.MATCH_HISTORY) 
        reader = csv.DictReader(f)

        for game in tqdm.tqdm(reader):
            if game[ELOConstants.OUTCOME] == ELOConstants.PLAYER1:
                winner = game[ELOConstants.PLAYER1]
                loser = game[ELOConstants.PLAYER2]
            elif game[ELOConstants.OUTCOME] == ELOConstants.PLAYER2:
                winner = game[ELOConstants.PLAYER2]
                loser = game[ELOConstants.PLAYER1]
            elif game[ELOConstants.OUTCOME] == ELOConstants.DRAW:
                winner = game[ELOConstants.PLAYER2]
                loser = game[ELOConstants.PLAYER1]

            draw = game[ELOConstants.OUTCOME] == ELOConstants.DRAW

            self.match(winner, loser, draw, record=False, print_result=False)

        f.close()


    def __create_match_history_file(self):
        with open(Files.MATCH_HISTORY, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ELOConstants.FIELDNAMES)
            writer.writeheader()

    def __win_expectation(self, rating, opponent_rating):
        return 1 / (1 + 10**((opponent_rating - rating)/400))

    def get_rating(self, id):
        if id in self.ratings:
            rating = self.ratings[id]
        else:
            stars = read_stars(id)
            if stars == 0:
                rating = 400
            if stars == 1:
                rating = 500
            elif stars == 2:
                rating = 600
            elif stars == 3:
                rating = 1000
            elif stars == 4:
                rating = 1400
            elif stars == 5:
                rating = 1800
            else:
                rating = 400

        return rating

    def match(self, winner, loser, draw=False, record=True, print_result=True):
        winner_rating = self.get_rating(winner)
        loser_rating = self.get_rating(loser)
        
        winner_expectation = self.__win_expectation(winner_rating, loser_rating)
        loser_expectation = self.__win_expectation(loser_rating, winner_rating)

        if draw:
            match_constant_winner = 0.5
            match_constant_loser = 0.5
        else:
            match_constant_winner = 1.0
            match_constant_loser = 0.0

        new_winner_rating = winner_rating + 32 * (match_constant_winner - winner_expectation)
        self.ratings[winner] = new_winner_rating

        new_loser_rating  = loser_rating + 32 * (match_constant_loser - loser_expectation)
        self.ratings[loser] = new_loser_rating


        if record:
            self.__record_match(winner, loser, draw)

        if print_result:
            print(f"Winner, {winner_rating} -> {new_winner_rating}")
            print(f"Loser, {loser_rating} -> {new_loser_rating}")

    def __record_match(self, winner, loser, draw):

        with open(Files.MATCH_HISTORY, "a") as f:
            writer = csv.DictWriter(f, ELOConstants.FIELDNAMES)

            if draw:
                outcome = ELOConstants.DRAW
            else:
                outcome = ELOConstants.PLAYER1

            elem = {ELOConstants.PLAYER1: winner,
                    ELOConstants.PLAYER2: loser,
                    ELOConstants.OUTCOME: outcome}
            writer.writerow(elem)

    def table(self):
        a = list(self.ratings.items())
        b = sorted(a, key=lambda x: x[1])
        pprint(b)

    def top(self, top_k):
        a = list(self.ratings.items())
        b = sorted(a, key=lambda x: x[1])
        top =  b[-top_k:]
        top.reverse()
        return top


img_loader = ImageLoader()
elo = ELO()

def update(text):
    if text=="right":
        winner_id = img_loader.rhs_id
        loser_id = img_loader.lhs_id

        draw = False

    elif text == "left":
        winner_id = img_loader.lhs_id
        loser_id = img_loader.rhs_id

        draw = False

    elif text == "draw":
        winner_id = img_loader.lhs_id
        loser_id = img_loader.rhs_id
        draw = True

    elo.match(winner_id, loser_id, draw)

    elo.table()

    
    new_rhs_image = img_loader.rhs()
    new_lhs_image = img_loader.lhs()

    return new_lhs_image, new_rhs_image

def refresh_images():
    lhs = img_loader.lhs()
    rhs = img_loader.rhs()
    return lhs, rhs

def multi_match(*args):
    print(args)
    return img_loader.lhs()

def clip_classify(search_string, subset_name):
    pu.db
    from PIL import Image
    import requests
    from transformers import AutoProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    search_strings = search_string.split(",")

    logging.info("Processing images")
    inputs = processor(
        text=search_strings, images=img_loader.all_images(), return_tensors="pt", padding=True
    )

    logging.info("Embedding images")
    outputs = model(**inputs)

    logging.info("Classifying images")
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    pu.db


with gr.Blocks() as demo:

    with gr.Tab("Create Subset"):
        search_string = gr.Textbox("Search string")
        subset_name = gr.Textbox("Subset name")
        button = gr.Button("Create subset")
        button.click(fn=clip_classify, inputs=[search_string, subset_name])

    with gr.Tab("Multi match"):
        with gr.Row():
            lhs = gr.Image(img_loader.lhs_fixed, height=HEIGHT, label="left")
            rhs_1 = gr.Image(img_loader.rhs, height=HEIGHT, label="right")
        with gr.Row():
             outcome_1 = gr.Radio(["Left", "Draw", "Right"])

        with gr.Row():
            gr.Image(img_loader.lhs_fixed, height=HEIGHT, label="left")
            rhs_2 = gr.Image(img_loader.rhs, height=HEIGHT, label="right")
        with gr.Row():
             outcome_2 = gr.Radio(["Left", "Draw", "Right"])
             

        button = gr.Button("Submit matches")
        button.click(fn=multi_match, inputs=[lhs, rhs_1, outcome_1, rhs_2, outcome_2])



    with gr.Tab("Single Match"):

        with gr.Row():
            lhs = gr.Image(img_loader.lhs, height=HEIGHT, label="left")
            rhs = gr.Image(img_loader.rhs, height=HEIGHT, label="right")
        with gr.Row():
            win_lhs = gr.Button("Left win")
            text_lhs = gr.Textbox("left", visible=False)
            win_lhs.click(fn=update, inputs=[text_lhs], outputs=[lhs, rhs])

            draw = gr.Button("Draw")
            text_draw = gr.Textbox("draw", visible=False)
            draw.click(fn=update, inputs=[text_lhs], outputs=[lhs, rhs])

            win_rhs = gr.Button("Right win")
            text_rhs = gr.Textbox("right", visible=False)
            win_rhs.click(fn=update, inputs=[text_rhs], outputs=[lhs, rhs])
        with gr.Row():
            refresh = gr.Button("Refresh")
            refresh.click(fn=refresh_images, outputs=[lhs, rhs]) 

    with gr.Tab("Rating Table"):
        top = elo.top(30)

        images = [(img_loader.read_image(path), rating) for path, rating in top]

        gr.Gallery(images,
                   columns=3,
                   rows=10,
                   object_fit="contain",
                   preview=False,
                   height="max-content")
        
demo.launch()
