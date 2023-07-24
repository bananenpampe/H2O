



# an idea of a composition transformer


# takes the frames (ie systems) and does something with them 


# implements a fit, transform, forward, inverse_forward and inverse_transform?

# inverse_forward is just a super() call to forward and applies the invers transform

# has to have a is_global attribute

import rascaline
import rascaline_torch
import numpy as np
import equistore
import equisolve
import torch
import sklearn.linear_model

#TODO: make an abstract class that defines common behaviours ie check_fitted etc.



# NOTE: THERE SHOULD BE NO BIAS IN THE LINEAR REGRESSION. IF THE COMPOSITION IS GLOBAL, 
# RISK IS THAT THE MODEL IS NOT SIZE DEPENDENT

class CompositionTransformer(torch.nn.Module):
    
    def __init__(self, bias=False):
        super().__init__()

        self.is_global = True
        
        self.calc = rascaline_torch.Calculator(
                                rascaline.AtomicComposition(per_structure=False)
                                )
        self.bias = bias
        self.is_fitted = False
        self.weights = None
        self.requires_features = False
        self.requires_systems = True
    
    def _check_fitted(self):
        assert self.is_fitted, "Transformer has to be fitted before calling transform or forward"
    
    def _compute_feat(self, systems):
        feats = self.calc(systems)
        feats = equistore.sum_over_samples(feats, "center")
        feats = feats.keys_to_properties("species_center")
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

        targets = targets.block(0).values
        targets = torch.clone(targets.reshape(-1,1))
        feats = torch.clone(feats)

        #print(feats.shape,targets.shape)
        weights, _, _, _ = torch.linalg.lstsq(feats, targets)

        return weights

    def forward(self, systems, targets):

        targets = targets.copy()
        self._check_fitted()
        # whats the issue here ? it uses to numpy
        
        return self.transform(systems, targets)
        
        #assume that targets already is a equistore.TensorMap ?


    def transform(self, systems, targets):

        targets = targets.copy()
        self._check_fitted()

        feats = self._compute_feat(systems)
        pred = feats @ self.weights

        target_values = targets.block(0).values
        target_values -= pred

        out_block = equistore.TensorBlock(values=target_values, 
                                          properties=targets.block(0).properties,
                                          components=targets.block(0).components,
                                          samples=targets.block(0).samples)



        for gradient_key, gradient in targets.block(0).gradients():
            out_block.add_gradient(gradient_key, gradient.copy())

        out_map = equistore.TensorMap(targets.keys, [out_block])

        return out_map
    
    def inverse_transform(self, systems, targets):
        self._check_fitted()


        #TODO: overly complicated ?
        # write a, get_offset function that inverse,
        #  and forward and transform can call

        targets = targets.copy()
        feats = self._compute_feat(systems)
        pred = feats @ self.weights

        target_values = targets.block(0).values
        target_values += pred

        out_block = equistore.TensorBlock(values=target_values, 
                                          properties=targets.block(0).properties,
                                          components=targets.block(0).components,
                                          samples=targets.block(0).samples)

        #out_map = equistore.TensorMap(targets.keys, [out_block])

        for gradient_key, gradient in targets.block(0).gradients():
            out_block.add_gradient(gradient_key, gradient.copy())

        out_map = equistore.TensorMap(targets.keys, [out_block])

        return out_map

        
    def fit(self, systems, targets):

        #solve a least squares problem using torch tensors
        feats = self._compute_feat(systems)
        self.weights = self._solve_weights(feats, targets)
        # do we have a bias ? -> concatenate ones to the features
        self.is_fitted = True

