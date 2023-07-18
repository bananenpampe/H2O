import torch
import equistore
from  .utils import l_to_str

class EquistoreLazyTorchApply(torch.nn.Module):
    """ Base class that sets up whatever torch.nn.Module 
    that takes as input a well-defined input and output dimension 
    and makes it compatible such that a TensorMap can be passed as input
    """

    def __init__(self, module: torch.nn.Module, n_out: int):
        super().__init__()
    
        self.m = module
        self.n_out = n_out
        self.m_map = torch.nn.ModuleDict({})
    
    def initialize_weights(self, input: equistore.TensorMap):
        
        for key, map in input.items():
            x = map.values
            key = l_to_str(key)
            self.m_map[key] = self.m(x.shape[1], self.n_out)
    
    def forward(self, input: equistore.TensorMap):
        
        out_blocks = []

        for key, block in input.items():

            key = l_to_str(key)
            
            out_values = self.m_map[key](block.values)
            out_samples = block.samples

            out_blocks.append(equistore.TensorBlock(values=out_values, 
                                              properties=equistore.Labels.range("property", out_values.shape[-1]), 
                                              components=[], 
                                              samples=out_samples))


        return equistore.TensorMap(input.keys, out_blocks)


    
    