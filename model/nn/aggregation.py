#TODO: write a mean aggregation Layer


import torch
import equistore
from functools import partial

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "equistore_torch_operations_futures"))

from reduce_over_samples import sum_over_samples, mean_over_samples
class StructureWiseAggregation(torch.nn.Module):
    
    def __init__(self,mode: str ="sum", sum_over: list = ["center"]):

        super().__init__()

        self.sum_over = sum_over
        self.mode = mode

        if mode not in ["mean", "sum", "passthrough"]:
            raise NotImplementedError(f"mode {mode} not implemented")

        if mode == "mean":
            self.aggregation_fn = partial(mean_over_samples,sample_names=self.sum_over)
        
        elif mode == "sum":
            self.aggregation_fn = partial(sum_over_samples,sample_names=self.sum_over)
        
        elif mode == "passthrough":
            self.aggregation_fn = lambda x: x
    
    def initialize_weights(self, inputs: equistore.TensorMap):
        pass

    def forward(self, inputs: equistore.TensorMap):
        return self.aggregation_fn(inputs)


class BPNNStructureWiseAggregation(StructureWiseAggregation):
    
    def __init__(self,mode: str ="sum"):

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

