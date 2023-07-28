import torch
from  .utils import l_to_str
from equistore.torch import TensorMap, TensorBlock, Labels


### There should be a GeneralTorchApply that accepts either a TensorMap or a dict of functions that 
### can be applied during application (there should be a unit apply)
# unit apply passes through

class EquistoreLazyTorchApply(torch.nn.Module):
    """ Base class that sets up whatever torch.nn.Module 
    that takes as input a well-defined input and output dimension 
    and makes it compatible such that a TensorMap can be passed as input
    """

    def __init__(self, 
                 module: torch.nn.Module, 
                 n_out: int,
                 property_str: str = "property") -> None:
        
        super().__init__()
    
        self.m = module
        self.n_out = n_out
        self.m_map = torch.nn.ModuleDict({})
        self.property_str = property_str
    
    def initialize_weights(self, input: TensorMap) -> None:

        #BETTER:
        # TODO: have a user-defined initialization-rule
        # a dict that maps from block_keys to initialization rules ?
        # -> a sample-size per block rule ?
        # smaller NN if less samples


        
        for key, block in input.items():

            assert block.components == [], "components are not supported yet"

            x = block.values
            key = str(key)
            self.m_map[key] = self.m(x.shape[1], self.n_out)
    
    def forward(self, input: TensorMap) -> TensorMap:
        
        out_blocks = []

        for key, block in input.items():

            key = str(key)
            
            out_values = self.m_map[key](block.values)
            out_samples = block.samples

            out_blocks.append(TensorBlock(values=out_values, 
                                              properties=Labels.range(self.property_str, 
                                              out_values.shape[-1]), 
                                              components=[], 
                                              samples=out_samples))


        return TensorMap(input.keys, out_blocks)


    
    