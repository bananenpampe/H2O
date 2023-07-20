#TODO: write a dummy-passthrough layer

import torch
import equistore

# Feature layers generate atomic-environment features.
# layers that either compute features (ie torch-spex)
# or layers that manipulate features from rascaline (ie alchemical combination)


class UnitFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    #every module should have this "prototype" function
    def initialize_weights(self, inputs: equistore.TensorMap):
        pass

    def forward(self, inputs: equistore.TensorMap ):
        return inputs


class DensifyFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: equistore.TensorMap ):

        inputs = inputs.keys_to_samples("species_center")

        return inputs