from pymilvus import Collection 
import io_utils
import db_utils
import pudb
import numpy as np

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
from sklearn.metrics import f1_score

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

    train_ratio = 0.8

    train_pos = int(len(positives) * train_ratio)
    val_pos = len(positives) - train_pos 

    train_neg = int(len(negatives) * train_ratio)
    val_neg = len(negatives) - train_neg 

    positives_split = torch.utils.data.random_split(dataset=positives, lengths=[train_pos, val_pos], generator=torch.Generator().manual_seed(42)) 
    negatives_split = torch.utils.data.random_split(negatives, [train_neg, val_neg], generator=torch.Generator().manual_seed(42)) 

    training_set = positives_split[0] + negatives_split[0]
    validation_set = positives_split[1] + negatives_split[1]
    return training_set, validation_set

def get_ffn():
    n_layers = 5
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

    targets = []
    try:
        targets = [[elem["target"]] for elem in batch]
        targets = torch.tensor(targets).to("cuda")
    except:
        pass

    vector_ids = [elem["vector_id"] for elem in batch]

    return vectors, targets, vector_ids

    
def train(label):
    training_set, validation_set = __load_splitted_data(label)

    ffn = get_ffn()

    bce_loss = torch.nn.BCELoss()
    params = ffn.parameters()
    optimizer = torch.optim.AdamW(params,
                                  lr=1e-09,
                                  weight_decay=0.1)

    epochs = 20
    for epoch_num in range(epochs):

        dataloader = DataLoader(training_set, collate_fn=__collate_fn, batch_size=8, shuffle=True)
        for vectors, targets, vector_ids in tqdm(dataloader, desc=f"Epoch: {epoch_num}"):
            output = ffn(vectors)

            loss = bce_loss(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    #Threshold
    dataloader = DataLoader(training_set, collate_fn=__collate_fn, batch_size=8, shuffle=True)
    all_probabilities = []
    all_targets = []
    for vectors, targets, vector_ids in tqdm(dataloader, desc="Threshold"):
        output = ffn(vectors)
        all_probabilities.extend(output.tolist())
        all_targets.extend(targets.tolist())
    
    best_score = 0.0
    best_threshold = 0.7
    # for threshold in np.arange(0.0, 0.99, 0.01):
        # outputs = [0 if prob < threshold else 1 for prob in all_probabilities]
        # score = f1_score(all_targets, outputs)
        # if best_score <= score:
            # best_threshold = threshold
            # best_score = score

    io_utils.save_threshold(label, best_threshold)
    io_utils.save_model(label, ffn)

def classify(label, threshold):
    dataset = db_utils.all_data()
    ffn = io_utils.load_model(label)
    # threshold = io_utils.load_threshold(label)

    result_vector_ids = []
    dataloader = DataLoader(dataset, collate_fn=__collate_fn, batch_size=8, shuffle=True)
    for vectors, targets, vector_ids in tqdm(dataloader):
        probabilities = ffn(vectors)

        outputs = [vector_id  for prob, vector_id in zip(probabilities, vector_ids) if threshold < prob]
        result_vector_ids.extend( outputs )

    return result_vector_ids



