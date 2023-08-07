



# an idea of a composition transformer


# takes the frames (ie systems) and does something with them 


# implements a fit, transform, forward, inverse_forward and inverse_transform?

# inverse_forward is just a super() call to forward and applies the invers transform

# has to have a is_global attribute

import rascaline
import rascaline.torch
import numpy as np
import equistore

from equistore.torch import TensorMap, TensorBlock

import torch

import sys 
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "equistore_torch_operations_futures"))

from reduce_over_samples import sum_over_samples, mean_over_samples

#TODO: make an abstract class that defines common behaviours ie check_fitted etc.



# NOTE: THERE SHOULD BE NO BIAS IN THE LINEAR REGRESSION. IF THE COMPOSITION IS GLOBAL, 
# RISK IS THAT THE MODEL IS NOT SIZE DEPENDENT

def get_system_global_composition(systems):
    # returns a torch tensor that contain the unique species in list of systems

    unique_species = []

    for syst in systems:
        # syst.species is a torch.tensor
        unique_species.extend(torch.unique(syst.species))
    
    unique_species = torch.unique(torch.stack(unique_species).flatten())

    return unique_species

class CompositionTransformer(torch.nn.Module):
    
    def __init__(self, bias=False):
        super().__init__()

        self.is_global = True
        
        self.calc = rascaline.torch.AtomicComposition(per_structure=False)

        self.bias = bias
        self.is_fitted = False
        self.weights = None
        self.requires_features = False
        self.requires_systems = True
        self.unique_species = None
        self.unique_labels = None
    
    def _check_fitted(self):
        assert self.is_fitted, "Transformer has to be fitted before calling transform or forward"
    
    def _compute_feat(self, systems):
        feats = self.calc(systems)
        feats = sum_over_samples(feats, "center")


        feats = feats.keys_to_properties(self.unique_labels)
        feats = feats.block(0).values

        # one element slice should preserve dimensionality
        if len(feats.shape) == 1:
            feats = feats.reshape(-1,1)
        
        if self.bias:
            return torch.cat([feats,torch.ones_like(feats[:,0:1])], dim=1)
        else:
            return feats

    def _solve_weights(self, feats, targets):
       
        # solve lstq problem with torch linalg.lstsq
        # feats have shape (n_samples, n_features)
        # targets have shape (n_samples, 1)

        #print(targets)

        targets = targets.block(0).values
        targets = torch.clone(targets.reshape(-1,1))
        feats = torch.clone(feats)

        #print(feats.shape,targets.shape)
        weights, _, _, _ = torch.linalg.lstsq(feats, targets)

        return weights

    def forward(self, systems, targets):

        assert self.weights.requires_grad == False

        targets = targets.copy()
        self._check_fitted()
        # whats the issue here ? it uses to numpy
        
        return self.transform(systems, targets)
        
        #assume that targets already is a equistore.TensorMap ?


    def transform(self, systems, targets):

        #assert self.weights.requires_grad == False

        targets = targets.copy()
        self._check_fitted()

        feats = self._compute_feat(systems)
        pred = feats @ self.weights

        target_values = targets.block(0).values
        target_values -= pred

        out_block = TensorBlock(values=target_values, 
                                          properties=targets.block(0).properties,
                                          components=targets.block(0).components,
                                          samples=targets.block(0).samples)

        for gradient_key in targets.block(0).gradients():
            gradient =  targets.block(0).gradient(gradient_key)
            out_block.add_gradient(gradient_key, gradient.copy())

        out_map = TensorMap(targets.keys, [out_block])

        return out_map
    
    def inverse_transform(self, systems, targets):
        self._check_fitted()

        #assert self.weights.requires_grad == False


        #TODO: overly complicated ?
        # write a, get_offset function that inverse,
        #  and forward and transform can call

        targets = targets.copy()
        feats = self._compute_feat(systems)
        pred = feats @ self.weights

        target_values = targets.block(0).values
        target_values += pred

        out_block = TensorBlock(values=target_values, 
                                          properties=targets.block(0).properties,
                                          components=targets.block(0).components,
                                          samples=targets.block(0).samples)

        #out_map = equistore.TensorMap(targets.keys, [out_block])

        for gradient_key in targets.block(0).gradients():
            gradient =  targets.block(0).gradient(gradient_key)
            out_block.add_gradient(gradient_key, gradient.copy())

        out_map = TensorMap(targets.keys, [out_block])

        return out_map

        
    def fit(self, systems, targets):

        #solve a least squares problem using torch tensors

        self.unique_species = get_system_global_composition(systems)
        unique_labels = torch.tensor(self.unique_species, dtype=torch.int32).reshape(-1,1)
        self.unique_labels = equistore.Labels(["species_center"], values=unique_labels)

        feats = self._compute_feat(systems)
        weights = self._solve_weights(feats, targets)
        self.weights = torch.nn.Parameter(weights, requires_grad=False)
        #self.weights.requires_grad = False
        # do we have a bias ? -> concatenate ones to the features
        self.is_fitted = True
    
    def setup(self, systems):
        assert self.weights is not None, "Transformer has to have weights loaded from statedict before calling setup"

        self.is_fitted = True
        self.unique_species = get_system_global_composition(systems)
        unique_labels = torch.tensor(self.unique_species, dtype=torch.int32).reshape(-1,1)
        self.unique_labels = equistore.Labels(["species_center"], values=unique_labels)


