
import torch
import metatensor

"""
class TensorMapMSE(torch.nn.Module):
    
    def __init__(self, loss, map):
        super().__init__()

    def forward(self, input, targets):
        return self.loss(input, targets)


class metatensorLoss(torch.nn.Module):
    #TODO: implement a general 
    pass
"""

#CPRS loss


def CRPS_func(means: torch.Tensor, targets: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
    """ Computes the CRPS of a gaussian distribution and a mean and uncertainty estimate

    means: torch.Tensor
        shape (N_samples, N_outputs)
    targets: torch.Tensor
        shape (N_samples, N_outputs)
    vars: torch.Tensor
        shape (N_samples, N_outputs)

    """

    sigma = torch.sqrt(vars)
    norm_x = ( targets - means)/sigma
    # torch.tensor(ndtr(norm_x.numpy())) 
    cdf =   0.5 * (1 + torch.erf(norm_x / torch.sqrt(torch.tensor(2))))

    normalization = 1 / (torch.sqrt(torch.tensor(2.0*torch.pi)))

    pdf = normalization * torch.exp(-(norm_x ** 2)/2.0)
    
    crps = sigma * (norm_x * (2*cdf-1) + 2 * pdf - 1/(torch.sqrt(torch.tensor(torch.pi))))

    return torch.mean(crps)

class CRPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, targets, vars):
        return CRPS_func(input, targets, vars)

class EnergyForceUncertaintyLoss(torch.nn.Module):

    def __init__(self,
                 w_forces: bool = True,
                 force_weight: float = 0.95,
                 base_loss: torch.nn.Module = torch.nn.GaussianNLLLoss,
                 force_loss: torch.nn.Module = torch.nn.MSELoss) -> None:
        
        super().__init__()

        force_weight = max(0.0, min(1.0, force_weight))

        if w_forces:
            self.force_weight = force_weight
            self.energy_weight = 1.0 - self.force_weight
        else:
            self.energy_weight = 1.0

        self.w_forces = w_forces
        self.energy_loss = base_loss()
        self.force_loss = force_loss()
    
    def report(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> dict:

        energy_pred_mean = input.block(0).values
        energy_pred_var = input.block(1).values
        energy_target = targets.block(0).values


        energy_nll = self.energy_loss(energy_pred_mean, energy_target, energy_pred_var) #input, target, var
        forces_nll = torch.tensor(0.0)

        if self.w_forces:
            
            forces_pred_mean = input.block(0).gradient("positions").values
            #forces_pred_var = input.block(1).gradient("positions").values
            forces_target = targets.block(0).gradient("positions").values
            
            forces_nll += self.force_loss(forces_pred_mean.flatten(),
                                          forces_target.flatten(),)
                                          #forces_pred_var.flatten())

        #not quite sure why I needed this
        energy_target.detach()

        return energy_nll, forces_nll


    def forward(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> torch.Tensor:
        
        #TODO: there are two ways how to solve this:
        # simple: get the torch tensors from the metatensor.TensorMap and use the torch.nn.MSELoss
        # write a metatensorLoss that takes two metatensor.TensorMaps and computes the (mse) loss
        
        energy_pred_mean = input.block(0).values
        energy_pred_var = input.block(1).values
        energy_target = targets.block(0).values

        #print(energy_pred_mean)
        #print(energy_pred_var)

        loss = self.energy_weight * self.energy_loss(energy_pred_mean.flatten(),
                                                     energy_target.flatten(),
                                                     energy_pred_var.flatten())


        if self.w_forces:
            forces_pred_mean = input.block(0).gradient("positions").values
            #forces_pred_var = input.block(1).gradient("positions").values

            #print(forces_pred_mean)
            #print(forces_pred_var)

            forces_target = targets.block(0).gradient("positions").values

            loss += self.force_weight * self.force_loss(forces_pred_mean.flatten(),
                                                        forces_target.flatten())
                                                        #forces_pred_var.flatten())

        energy_target.detach()

        return loss

class EnergyForceLoss(torch.nn.Module):

    def __init__(self,
                 w_forces: bool = True,
                 force_weight: float = 0.95,
                 base_loss: torch.nn.Module = torch.nn.MSELoss) -> None:
        
        super().__init__()

        force_weight = max(0.0, min(1.0, force_weight))

        if w_forces:
            self.force_weight = force_weight
            self.energy_weight = 1.0 - self.force_weight
        else:
            self.energy_weight = 1.0

        self.w_forces = w_forces
        self.energy_loss = base_loss()
        self.force_loss = base_loss()
    
    def report(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> dict:

        energy_pred = input.block(0).values
        energy_target = targets.block(0).values

        energy_mse = self.energy_loss(energy_pred, energy_target)
        forces_mse = torch.tensor(0.0)

        if self.w_forces:
            forces_pred = input.block(0).gradient("positions").values
            forces_target = targets.block(0).gradient("positions").values
            forces_mse += self.force_loss(forces_pred.flatten(), forces_target.flatten())

        energy_target.detach()

        return energy_mse, forces_mse


    def forward(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> torch.Tensor:
        
        #TODO: there are two ways how to solve this:
        # simple: get the torch tensors from the metatensor.TensorMap and use the torch.nn.MSELoss
        # write a metatensorLoss that takes two metatensor.TensorMaps and computes the (mse) loss
        
        energy_pred = input.block(0).values
        energy_target = targets.block(0).values

        loss = self.energy_weight * self.energy_loss(energy_pred, energy_target) 

        if self.w_forces:
            forces_pred = input.block(0).gradient("positions").values
            forces_target = targets.block(0).gradient("positions").values

            loss += self.force_weight * self.force_loss(forces_pred.flatten(), forces_target.flatten())

        energy_target.detach()

        return loss


class GeneralLossUQ(torch.nn.Module):

    def __init__(self,
                 w_forces: bool = True,
                 force_weight: float = 0.95,
                 base_loss: torch.nn.Module = torch.nn.GaussianNLLLoss) -> None:
        
        super().__init__()

        self.loss_fn = base_loss()
    
    def forward(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> dict:

        pred = input.block(0).values
        pred_var = input.block(1).values
        target = targets.block(0).values

        loss_ = self.loss_fn(pred.flatten(), target.flatten(), pred_var.flatten())

        return loss_


    def report(self, input: metatensor.TensorMap, targets: metatensor.TensorMap) -> dict:
        return self.forward(input, targets)
