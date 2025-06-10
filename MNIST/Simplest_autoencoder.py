from typing import List
import torch
from torch.nn import Linear, ModuleList, Sigmoid
from torch.nn import Module

class Autoencoder(Module):
    def __init__(self, dimensions):
        super().__init__()

        dim_encoder = dimensions[:len(dimensions)//2+1]
        dim_decoder = dimensions[len(dimensions)//2:]

        self.check_valid_dims(dim_encoder, dim_decoder)

        self.encoder = Encoder(dim_encoder)
        self.decoder = Decoder(dim_decoder)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x

    def check_valid_dims(self, dim_encoder, dim_decoder):
        assert len(dim_encoder) == len(dim_decoder)
        assert len(dim_encoder) > 0
        assert len(dim_decoder) > 0

class Encoder(Module):

    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.layers = ModuleList([Linear(self.dimensions[i], self.dimensions[i+1]) for i in range(len(self.dimensions)-1)])
        self.activation = Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

class Decoder(Module):

    def __init__(self, dimensions : List):
        super().__init__()
        self.dimensions = dimensions
        self.layers = ModuleList([Linear(self.dimensions[i], self.dimensions[i+1]) for i in range(len(self.dimensions)-1)])
        self.activation = Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

if __name__ == "__main__":
    a = Autoencoder([758, 64, 758])
    a.forward(torch.randn(758))
