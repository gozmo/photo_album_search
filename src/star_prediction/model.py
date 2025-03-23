from torch import nn
from transformers import ResNetModel

class Model(nn.Module):
    def __init__(self, device, rating_mapping):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        self.dense = Dense(len(rating_mapping)).to(device)

    def forward(self, images):
        resnet_output = self.resnet(**images)
        prediction = self.dense(resnet_output.last_hidden_state)
        return prediction


class Dense(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.layers = nn.Sequential(
                                    nn.Dropout(0.5),
                                    nn.LayerNorm((2048*7*7)),
                                    nn.Linear(2048*7*7, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, output_size),
                                    nn.Softmax(dim=0))

    def forward(self, x):
        bs, _, _ ,_ = x.shape
        x_ = x.reshape(bs, 2048*7*7)
        output = self.layers(x_)
        return output
