from pymilvus import Collection 
import io_utils
import db_utils

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
    
    for elem in elems:
        vector_id = int(elem["vector_id"])
        elem["target"] = [1.0 if annotations[vector_id] == "True" else 0.0]

    return elems

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

    targets = [elem["target"] for elem in batch]
    targets = torch.tensor(targets).to("cuda")

    return vectors, targets

    
def train(label):
    dataset = __load_training_set(label)
    ffn = get_ffn()

    bce_loss = torch.nn.BCELoss()
    params = ffn.parameters()
    optimizer = torch.optim.AdamW(params,
                                  lr=1e-05,
                                  weight_decay=0.1)

    epochs = 5
    for epoch_num in range(epochs):
        print(f"Epoch:{epoch_num}")

        dataloader = DataLoader(dataset, collate_fn=__collate_fn, batch_size=8, shuffle=True)
        for vectors, targets in dataloader:
            output = ffn(vectors)

            loss = bce_loss(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def classify(label):
    pass
