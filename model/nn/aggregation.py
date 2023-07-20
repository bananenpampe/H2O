#TODO: write a mean aggregation Layer


import torch
import equistore
from functools import partial

class StructureWiseAggregation(torch.nn.Module):
    
    def __init__(self,mode: str ="mean", sum_over: list = ["center"]):

        super().__init__()

        self.sum_over = sum_over

        if mode not in ["mean", "sum", "passthrough"]:
            raise NotImplementedError(f"mode {mode} not implemented")

        if mode == "mean":
            self.aggregation_fn = partial(equistore.mean_over_samples,sample_names=self.sum_over)
        
        elif mode == "sum":
            self.aggregation_fn = partial(equistore.sum_over_samples,sample_names=self.sum_over)
        
        elif mode == "passthrough":
            self.aggregation_fn = lambda x: x
    
    def initialize_weights(self, inputs: equistore.TensorMap):
        pass

    def forward(self, inputs: equistore.TensorMap):
        return self.aggregation_fn(inputs)


class BPNNStructureWiseAggregation(StructureWiseAggregation):
    
    def __init__(self,mode: str ="mean"):

        super().__init__(mode=mode, sum_over=["species_center","center"])
    
    def forward(self, inputs: equistore.TensorMap):
        #print(inputs.block(0).samples)
        inputs = inputs.keys_to_samples("species_center")
        return self.aggregation_fn(inputs)



"""

class WeightedAggregation(StructureWiseAggregation):

    def forward(self, inputs: equistore.TensorMap, positions: torch.Tensor):
        #TODO: implement this
        # -> the dipole moment of a series of predicted
        #inputs = inputs.keys_to_samples("species_center")
        return self.aggregation_fn(inputs, positions)
    
"""

