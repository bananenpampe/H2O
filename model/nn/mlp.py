# TODO: write a generic MLP, takes n_hidden_layers, n_in, n_out, activation 
# and number of hidden neurons as either int, or tuple (64,64,32)
# gets metatensor TensorMap and returns metatensor TensorMap
# TODO: write a block-wise mlp, that applies an MLP to each TensorBlock of the input
# either takes n_in as fixed across all blocks, or as a tuple (n_1,n_2,n_3) ?
# gets metatensor TensorMap and returns metatensor TensorMap
# TODO: write a BPNN interaction that assumes that block_keys are center species

import torch

class MLP_mean(torch.nn.Module):
    """ Initializes a mean MLP with an arbitrary number of hidden layers and activation functions
    """
    def __init__(self,
                 n_in: int, 
                 n_out: int,
                 n_hidden: int = 32,
                 n_hidden_layers: int = 2,
                 activation: torch.nn.Module = torch.nn.SiLU,
                 w_bias: bool = True,
                 mve: bool = False) -> None:
        
        """
        Args:
            n_in (int): number of input features
            n_out (int): number of output features
            n_hidden (int): number of hidden neurons
            n_hidden_layers (int): number of hidden layers
            activation (torch.nn.module): activation function
        """
        
        super().__init__()

        self.mve = mve

        if self.mve:
            assert n_out % 2 == 0, "n_out must be 2 for mve"
            assert w_bias == False, "bias must be False for mve"

        modules = []

        if n_hidden_layers > 0:
            modules.append(torch.nn.Linear(n_in,n_hidden))
            modules.append(activation())
        else:
            modules.append(torch.nn.Linear(n_in,n_hidden))
        
        
        for n in range(n_hidden_layers-1):
            modules.append(torch.nn.Linear(n_hidden,n_hidden))
            modules.append(activation())
        
        self.nn = torch.nn.Sequential(
                *modules
                )
        
        self.mean_out = torch.nn.Linear(n_hidden,n_out, bias=w_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_hidden = self.nn(x)
        mean = self.mean_out(x_hidden)

        if self.mve:
            mean_out = mean[:,0]
            var = torch.nn.functional.softplus(mean[:,1])
            mean = torch.stack((mean_out, var), dim=1)
        
        return mean


class EnsembleMLP(torch.nn.Module):
    """ Initializes an ensemble of MLPs with an arbitrary number of hidden layers and activation functions
    overwrites the n_out keyword of shallow ensembles to the number of committee members
    """
    def __init__(self,
                 n_in: int, 
                 n_out: int,
                 n_hidden: int = 32,
                 n_hidden_layers: int = 2,
                 activation: torch.nn.Module = torch.nn.SiLU,
                 w_bias: bool = True,
                 mve: bool = False) -> None:
        
        """
        Args:
            n_in (int): number of input features
            n_out (int): number of output features
            n_hidden (int): number of hidden neurons
            n_hidden_layers (int): number of hidden layers
            activation (torch.nn.module): activation function
            n_ensemble (int): number of ensemble members
        """
        
        super().__init__()

        self.mve = mve



        if self.mve:
            assert n_out % 2 == 0, "n_out must be 2 for mve"
            assert w_bias == False, "bias must be False for mve"
            raise NotImplementedError("MVE not implemented yet")

        self.ensemble = torch.nn.ModuleList([MLP_mean(n_in, 1, n_hidden, n_hidden_layers, activation, w_bias, mve) for _ in range(n_out)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        mean = torch.cat([model(x) for model in self.ensemble], dim=1)

        return mean