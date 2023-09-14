
from .modules import metatensorLazyTorchApply
from .mlp import MLP_mean
import torch

class metatensorMLPLazy(metatensorLazyTorchApply):
    """ Wrapper class that sets up a predefined MLP
    that then gets applied block wise
    """
    
    def __init__(self, n_out: int,
                 n_hidden: int = 32,
                 n_hidden_layers: int = 2,
                 activation: torch.nn.Module = torch.nn.Tanh):
        
        def predifined_MLP_factory(n_in: int, n_out: int):
            """ Returns a predefined MLP
            """
            return MLP_mean(n_in=n_in,
                            n_out=n_out,
                            n_hidden=n_hidden,
                            n_hidden_layers=n_hidden_layers,
                            activation=activation)
        
        super().__init__(predifined_MLP_factory, n_out)