#TODO: write a dummy-passthrough layer

import torch
import metatensor

# Feature layers generate atomic-environment features.
# layers that either compute features (ie torch-spex)
# or layers that manipulate features from rascaline (ie alchemical combination)


class UnitFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    #every module should have this "prototype" function
    def initialize_weights(self, inputs: metatensor.TensorMap):
        pass

    def forward(self, inputs: metatensor.TensorMap ):
        return inputs

"""

# could, in principle handle the densification

class BPNNFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    #every module should have this "prototype" function
    def initialize_weights(self, inputs: metatensor.TensorMap):
        pass

    def forward(self, inputs: metatensor.TensorMap ):
        
    # to keys_to_neighbours here
        return inputs
"""

class DensifyFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: metatensor.TensorMap ):

        inputs = inputs.keys_to_samples("species_center")

        return inputs