#TODO: write a mean aggregation Layer


import torch
import metatensor
from functools import partial

import sys
import os

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "metatensor_torch_operations_futures"))

from metatensor.torch import sum_over_samples, mean_over_samples
from metatensor.torch import TensorBlock, Labels, TensorMap
class StructureWiseAggregation(torch.nn.Module):
    
    """Class that aggregates atomistic predictions.
    Modes implemented:
    - sum: sum over all atomic contributions
    - mean: mean over all atomic contributions
    - passthrough: no aggregation, just return the input

    """
    
    def __init__(self,mode: str ="sum", sum_over: list = ["center"]):
        """
        Args:
            mode: aggregation mode
            sum_over: list of keys to sum over
        """

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
    
    def initialize_weights(self, inputs: metatensor.TensorMap):
        pass

    def forward(self, inputs: metatensor.TensorMap):
        return self.aggregation_fn(inputs)



class BPNNStructureWiseAggregation(StructureWiseAggregation):
    
    def __init__(self,mode: str ="sum"):

        super().__init__(mode=mode, sum_over=["species_center","center"])
    
    def forward(self, inputs: metatensor.TensorMap):
        #print(inputs.block(0).samples)
        inputs = inputs.keys_to_samples("species_center")
        return self.aggregation_fn(inputs)

class BPNNStructureWiseAggregationVar(StructureWiseAggregation):
    
    def __init__(self, mode: str ="sum", use_shallow_ensemble=True, predict_std_err=False):

        self.use_shallow_ensemble = use_shallow_ensemble
        self.predict_std_err = predict_std_err

        super().__init__(mode=mode, sum_over=["species_center","center"])
    
    def forward(self, inputs: metatensor.TensorMap):
        #print(inputs.block(0).samples)

        
        inputs = inputs.keys_to_samples("species_center")
        

        #should return a tensormap of values with shape (N_structures, N_ensembles)
        out = self.aggregation_fn(inputs)

        name_0 = out.block(0).properties.names[0]
        val_0 = out.block(0).properties.values[0].reshape(-1,1)

        if self.use_shallow_ensemble:
            mean_in = torch.mean(out.block(0).values, dim=1)
            var_in = torch.var(out.block(0).values, dim=1)

        else:
            
            mean_in = out.block(0).values[:,0]
            var_in = out.block(0).values[:,1]
            
            if self.predict_std_err:
                var_in = var_in ** 2


        block_0 = TensorBlock(values=mean_in.reshape(-1, 1),
                                samples=out.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))

        
        block_1 = TensorBlock(values=var_in.reshape(-1, 1),
                                samples=out.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))
        
        #block_2 = input.block(0).copy()

        keys_in = Labels(["energy"], values=torch.tensor([0, 1]).reshape(-1,1))

        #add block_2 here later 

        return TensorMap(keys_in, [block_0, block_1])    


class BPNNAtomWiseAggregationVar(StructureWiseAggregation):
    
    def __init__(self, mode: str ="sum", use_shallow_ensemble=True, predict_std_err=False):

        self.use_shallow_ensemble = use_shallow_ensemble
        self.predict_std_err = predict_std_err

        super().__init__(mode=mode, sum_over=["species_center","center"])
    
    def forward(self, inputs: metatensor.TensorMap):
        #print(inputs.block(0).samples)
        #print(inputs.block(0).values.shape)
        #out = inputs.keys_to_samples("species_center")
        #print(inputs.block(0).values.shape)
        #should return a tensormap of values with shape (N_structures, N_ensembles)
        #out = self.aggregation_fn(inputs)
        #print(inputs.block(0).samples.names)
        #print(inputs.block(0).samples.values)
        out = inputs

        name_0 = out.block(0).properties.names[0]
        val_0 = out.block(0).properties.values[0].reshape(-1,1)

        if self.use_shallow_ensemble:
            mean_in = torch.mean(out.block(0).values, dim=1, keepdim=True)
            var_in = torch.var(out.block(0).values, dim=1, keepdim=True)
            #print(mean_in.shape)
            #print(var_in.shape)

        else:
            
            mean_in = out.block(0).values[:,0]
            var_in = out.block(0).values[:,1]
            
            if self.predict_std_err:
                var_in = var_in ** 2


        block_0 = TensorBlock(values=mean_in,
                                samples=out.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))

        
        block_1 = TensorBlock(values=var_in,
                                samples=out.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))
        
        #block_2 = input.block(0).copy()

        keys_in = Labels(["properties"], values=torch.tensor([0, 1]).reshape(-1,1))

        #add block_2 here later 

        return TensorMap(keys_in, [block_0, block_1])  

"""

class WeightedAggregation(StructureWiseAggregation):

    def forward(self, inputs: metatensor.TensorMap, positions: torch.Tensor):
        #TODO: implement this
        # -> the dipole moment of a series of predicted
        #inputs = inputs.keys_to_samples("species_center")
        return self.aggregation_fn(inputs, positions)
    
"""

