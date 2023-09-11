
#model contains

# feature -> interaction -> aggregation -> response
# how does schnet do it ?
# sequentially apply a number of transformations to the input data
# ie two output layers:
# energy layer writes energy metadata to the output -> forces layer than uses the metadata to get response

# TODO:
# write a model class

import torch
import metatensor
from .feature import UnitFeatures
from .interaction import BPNNInteraction
from .aggregation import StructureWiseAggregation, BPNNStructureWiseAggregation
from .response import UnitResponse, ForceRespone

class BPNNModel(torch.nn.Module):

    def __init__(self, 
                 feature: torch.nn.Module = UnitFeatures(),
                 interaction: torch.nn.Module = BPNNInteraction(n_out=1),
                 aggregation: torch.nn.Module = BPNNStructureWiseAggregation(mode="sum"),
                 response: torch.nn.Module = ForceRespone()):
    
        super().__init__()
        
        self.feature = feature
        self.interaction = interaction
        self.aggregation = aggregation
        self.response = response

    def initialize_weights(self, inputs: metatensor.TensorMap):
    
        self.feature.initialize_weights(inputs)
        inputs = self.feature(inputs)
        
        self.interaction.initialize_weights(inputs)
        inputs = self.interaction(inputs)
        
        self.aggregation.initialize_weights(inputs)
        inputs = self.aggregation(inputs)
        
        self.response.initialize_weights(inputs)
        #inputs = self.response(inputs)

    def forward(self, inputs, systems):
        
        features = self.feature(inputs)
        interactions = self.interaction(features)
        aggregations = self.aggregation(interactions)
        responses = self.response(aggregations, systems)
        
        return responses
    
    def get_energy(self, inputs, systems):
        #for uncertainty model, for now ;)
        
        features = self.feature(inputs)
        interactions = self.interaction(features)
        aggregations = self.aggregation(interactions)
        
        return aggregations
    