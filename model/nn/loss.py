
import torch
import equistore

"""
class TensorMapMSE(torch.nn.Module):
    
    def __init__(self, loss, map):
        super().__init__()

    def forward(self, input, targets):
        return self.loss(input, targets)


class EquistoreLoss(torch.nn.Module):
    #TODO: implement a general 
    pass
"""

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
    
    def report(self, input: equistore.TensorMap, targets: equistore.TensorMap) -> dict:

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


    def forward(self, input: equistore.TensorMap, targets: equistore.TensorMap) -> torch.Tensor:
        
        #TODO: there are two ways how to solve this:
        # simple: get the torch tensors from the equistore.TensorMap and use the torch.nn.MSELoss
        # write a equistoreLoss that takes two equistore.TensorMaps and computes the (mse) loss
        
        energy_pred = input.block(0).values
        energy_target = targets.block(0).values

        loss = self.energy_weight * self.energy_loss(energy_pred, energy_target) 

        if self.w_forces:
            forces_pred = input.block(0).gradient("positions").values
            forces_target = targets.block(0).gradient("positions").values

            loss += self.force_weight * self.force_loss(forces_pred.flatten(), forces_target.flatten())

        energy_target.detach()

        return loss


