from pymilvus import Collection 
import io_utils
import db_utils
import pudb

from tqdm import tqdm
import torch
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import LeakyReLU 
from torch.nn import Sigmoid
from torch.nn import Sequential
from torch.utils.data import DataLoader 
from collections import OrderedDict

def __load_training_set(label):
    annotations = io_utils.load_annotations(label)
    collection = db_utils.get_collection()

    vector_ids = list(annotations.keys())

    elems = db_utils.get_database_elems(vector_ids)
    
    positives = []
    negatives = []
    for elem in elems:
        vector_id = int(elem["vector_id"])
        if annotations[vector_id] == "True":
            elem["target"] = 1.0 
            positives.append(elem)
        else:
            elem["target"] = 0.0 
            negatives.append(elem)

    return positives, negatives

def __load_splitted_data(label):
    positives, negatives = __load_training_set(label)

    split_ratio = [0.8, 0.2]
    positives_split = torch.utils.data.random_split(positives, split_ratio) 
    negatives_split = torch.utils.data.random_split(negatives, split_ratio) 

    training_set = positives_split[0] + negatives_split[0]
    validation_set = positives_split[1] + negatives_split[1]
    return training_set, validation_set

def get_ffn():
    n_layers = 3
    dim_hidden = 300
    dim_input = 512
    dim_output = 1
    model_sequence = []
    model_sequence.append(("Dropout", Dropout(0.5)))
    model_sequence.append(("BatchNorm", BatchNorm1d(dim_input)))
    model_sequence.append(("Linear1", Linear(in_features=dim_input, out_features=dim_hidden)))
    model_sequence.append(("LeakyRelu", LeakyReLU(0.01, inplace=True)))

    for n in range(1, n_layers):
        model_sequence.append((f"Linear{n+1}", Linear(in_features=dim_hidden, out_features=dim_hidden)))
        model_sequence.append(("LeakyRelu", LeakyReLU(0.01, inplace=True)))
    model_sequence.append(("OutputLinear", Linear(in_features=dim_hidden, out_features=dim_output)))
    model_sequence.append(("sigmoid", Sigmoid()))
    ffn_model = Sequential(OrderedDict(model_sequence))

    return ffn_model.to("cuda")

def __collate_fn(batch):
    vectors = [elem["vector"] for elem in batch]
    vectors = torch.tensor(vectors).to("cuda")

    targets = [[elem["target"]] for elem in batch]
    targets = torch.tensor(targets).to("cuda")

    return vectors, targets

    
def train(label):
    training_set, validation_set = __load_splitted_data(label)

    ffn = get_ffn()

    bce_loss = torch.nn.BCELoss()
    params = ffn.parameters()
    optimizer = torch.optim.AdamW(params,
                                  lr=1e-05,
                                  weight_decay=0.1)

    epochs = 5
    for epoch_num in range(epochs):

        dataloader = DataLoader(training_set, collate_fn=__collate_fn, batch_size=8, shuffle=True)
        for vectors, targets in tqdm(dataloader, desc=f"Epoch: {epoch_num}"):
            output = ffn(vectors)

            loss = bce_loss(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    io_utils.save_model(label, ffn)

def classify(label):
    data = db_utils.all_data()
    model = io_utils.load_model(label)

    dataloader = DataLoader(dataset, collate_fn=__collate_fn, batch_size=8, shuffle=True)
    for vectors, targets in dataloader:
        output = ffn(vectors)
        #thresholds here


