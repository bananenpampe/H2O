# TODO: write a generic interaction layer
# TODO: beginn with a simple linear layer

import torch
import equistore
from .linear import EquistoreLinearLazy
from .nonlinear import EquistoreMLPLazy


class Interaction(torch.nn.Module):
    #better to have an abstract class

    def __init__(self, n_out):
        super().__init__()
        self.interaction = None
    
    def forward(self, inputs: equistore.TensorMap):
        return self.interaction(inputs)

class LinearInteraction(torch.nn.Module):
    
    def __init__(self, n_in, n_out):
        super().__init__()
        self.interaction = EquistoreLinearLazy(n_out)
    
    def initialize_weights(self, inputs: equistore.TensorMap):
        self.model.initialize_weights(inputs)

    def forward(self, inputs: equistore.TensorMap):
        return self.interaction(inputs)

class BPNNInteraction(torch.nn.Module):
    
    def __init__(self, n_out: int,
                 n_hidden: int = 32,
                 n_hidden_layers: int = 2,
                 activation: torch.nn.Module = torch.nn.Tanh):
        
        super().__init__()
        
        self.model = EquistoreMLPLazy(n_out, 
                                      n_hidden, 
                                      n_hidden_layers, 
                                      activation)

    def initialize_weights(self, inputs: equistore.TensorMap):
        self.model.initialize_weights(inputs)
    
    def forward(self, inputs: equistore.TensorMap):
        return self.model(inputs)