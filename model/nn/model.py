
#model contains

# feature -> interaction -> aggregation -> response
# how does schnet do it ?
# sequentially apply a number of transformations to the input data
# ie two output layers:
# energy layer writes energy metadata to the output -> forces layer than uses the metadata to get response

# TODO:
# write a model class

import torch
import copy
import metatensor
from metatensor.torch import sum_over_samples
from .feature import UnitFeatures
from .interaction import BPNNInteraction
from .aggregation import StructureWiseAggregation, BPNNStructureWiseAggregation
from .response import UnitResponse, ForceRespone

class BPNNModel(torch.nn.Module):
    """ Class that combines feture, interaction, aggregation and response layers in a BPNN-like model.

    """

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
        """Initializes the BPNN using an example TensorMap.
        MUST CONTAIN ALL THE EXPECTED KEYS FOR THE ENTIRE TRAINING SET/ USE CASES
        """
    
        self.feature.initialize_weights(inputs)
        inputs = self.feature(inputs)
        
        self.interaction.initialize_weights(inputs)
        inputs = self.interaction(inputs)
        
        self.aggregation.initialize_weights(inputs)
        inputs = self.aggregation(inputs)
        
        self.response.initialize_weights(inputs)
        #inputs = self.response(inputs)

    def forward(self, inputs, systems):
        """Forward pass of the BPNN model.

        Args:
            inputs: TensorMap containing the input features/data
            systems: List of torch.systems containing the positions etc.
        """
        
        features = self.feature(inputs)
        interactions = self.interaction(features)
        aggregations = self.aggregation(interactions)
        responses = self.response(aggregations, systems)
        
        return responses
    
    def get_atomic_energies(self, inputs, systems):
        """ Returns the atomic energies from the model.

        Args:
            inputs: TensorMap containing the input features/data
            systems: List of torch.systems containing the positions etc.

        """
        
        features = self.feature(inputs)
        interactions = self.interaction(features)

        #we need to call a keys_to_samples here ?
        return interactions.keys_to_samples("species_center")
    
    def get_hidden_features(self, 
                            inputs, 
                            systems, 
                            structure_wise=False):
        """ Returns the hidden features from the model.

        Args:
            inputs: TensorMap containing the input features/data
            systems: List of torch.systems containing the positions etc.

        """
        
        features = self.feature(inputs)

        # make a copy of the model and strip away the last layer
        int_model = copy.deepcopy(self.interaction)
        #interaction is a module dict
        # loop over modules in module dict and then remove last layer from each of the modules
        
        for key, module in int_model.model.m_map.items():
            int_model.model.m_map[key] = module.nn[:-1]
        
        interactions = int_model(features)

        if structure_wise is True:
            interactions = sum_over_samples(interactions, samples_names=["center"])
            interactions = interactions.keys_to_properties("species_center")
        
        else:
            interactions = interactions.keys_to_samples("species_center")

        return interactions
    
    def get_energy(self, inputs, systems):
        """ Returns the structure wise energies from the model.

        Args:
            inputs: TensorMap containing the input features/data
            systems: List of torch.systems containing the positions etc.

        """

        #for uncertainty model, for now ;)
        
        features = self.feature(inputs)
        interactions = self.interaction(features)
        aggregations = self.aggregation(interactions)
        
        return aggregations
    
    def get_committee_forces(self, inputs, systems):
        """ Returns the forces from the model.
        """
        #For now: lets be super lazy and just loop over samples
        #energies has dimensions (n_samples, n_species)
        # Iterate over each scalar energy to compute its gradient

        Energies = self.get_energy(inputs, systems).block(0).values

        #mean of forces equals the derivative of the 

        #mean of forces equals the derivative of the 

        grads_tot = []

        for  syst_i, E_sys in zip(systems, Energies):

            grads = []
            # Iterate over each scalar energy to compute its gradient
            
            for e_i in E_sys:
                # Zero-out previous gradients if any
                if syst_i.positions.grad is not None:
                    syst_i.positions.grad.zero_()

                # Compute gradient of e_i with respect to the positions
                e_i.backward(retain_graph=True)

                # Append the computed gradient to our list
                grads.append(syst_i.positions.grad.clone())

            grads = torch.stack(grads)
            grads_tot.append(grads)

        grads_tot = torch.cat(grads_tot, dim=1)

        return grads_tot

        

    