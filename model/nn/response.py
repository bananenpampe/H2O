import torch
import metatensor
import rascaline.torch
import numpy as np
from torch.autograd import grad
from typing import List

from metatensor.torch import TensorMap, TensorBlock, Labels

class UnitResponse(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # the unitresponse does nothing
    def initialize_weights(self, input: TensorMap):
        pass
    
    def forward(self, input: TensorMap, systems: List[rascaline.torch.System]) -> TensorMap:
        return input


class ForceUncertaintyRespone(UnitResponse):

    def forward(self, 
                input: TensorMap, 
                systems: List[rascaline.torch.System]) -> TensorMap:
        
        # input should be a tensormap of shape (n_structures, n_ensembles)

        name_0 = input.block(0).properties.names[0]
        val_0 = input.block(0).properties.values[0].reshape(-1,1)

        mean_in = torch.mean(input.block(0).values, dim=1)
        var_in = torch.var(input.block(0).values, dim=1)

        outputs_mean = list(torch.ones_like(mean_in))
        outputs_var = list(torch.ones_like(var_in))
        # TODO: we can simply do mean and variance here?

        dEdX = grad(outputs=list(mean_in),
                      inputs=[sys_i.positions for sys_i in systems],
                      grad_outputs=outputs_mean,
                      create_graph=True,
                      retain_graph=True)
        
        """
        dsigmadX = grad(outputs=list(var_in),
                      inputs=[sys_i.positions for sys_i in systems],
                      grad_outputs=outputs_var,
                      create_graph=True,
                      retain_graph=True)
        """
        #print("hello")

        #negative forces, are position gradients
        gradient_values = torch.vstack(dEdX)
        gradient_uncertainty = gradient_values

        position_gradient_samples = Labels(
            ["sample", "structure", "atom"],
            torch.tensor(np.array(
                [
                    [s, s, a]
                    for s in range(len(systems))
                    for a in range(len(dEdX[s]))
                ]
            ))
        )

        #TODO: change to torch labels once move to rascaline.rascaline-torch
        positions_gradient = TensorBlock(
            values=gradient_values.reshape(-1, 3, 1),
            samples=position_gradient_samples,
            components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
            properties=Labels([name_0], val_0),
        )

        
        positions_gradient_uncertainty = TensorBlock(
            values=gradient_uncertainty.reshape(-1, 3, 1),
            samples=position_gradient_samples,
            components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
            properties=Labels([name_0], val_0),
        )
        


        block_0 = TensorBlock(values=mean_in.reshape(-1, 1),
                                samples=input.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))
        
        block_0.add_gradient("positions", positions_gradient)

        
        block_1 = TensorBlock(values=var_in.reshape(-1, 1),
                                samples=input.block(0).samples,
                                components=[],
                                properties=Labels([name_0], val_0))
        
        block_1.add_gradient("positions", positions_gradient_uncertainty)
        

        #block_2 = input.block(0).copy()

        keys_in = Labels(["energy"], values=torch.tensor([0, 1]).reshape(-1,1))

        #add block_2 here later 

        return TensorMap(keys_in, [block_0, block_1])


class ForceRespone(UnitResponse):

    def forward(self, 
                input: TensorMap, 
                systems: List[rascaline.torch.System]) -> TensorMap:

        outputs = list(torch.ones_like(input.block(0).values))

        dEdx = grad(outputs=list(input.block(0).values),
                      inputs=[sys_i.positions for sys_i in systems],
                      grad_outputs=outputs,
                      create_graph=True,
                      retain_graph=True)
        
        #print("hello")

        #negative forces, are position gradients
        gradient_values = torch.vstack(dEdx)


        position_gradient_samples = Labels(
            ["sample", "structure", "atom"],
            torch.tensor(np.array(
                [
                    [s, s, a]
                    for s in range(len(systems))
                    for a in range(len(dEdx[s]))
                ]
            ))
        )

        #TODO: change to torch labels once move to rascaline.rascaline-torch
        positions_gradient = TensorBlock(
            values=gradient_values.reshape(-1, 3, 1),
            samples=position_gradient_samples,
            components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
            properties=input.block(0).properties,
        )

        block_c = input.block(0).copy()
        block_c.add_gradient("positions", positions_gradient)

        #print(input.keys)

        return TensorMap(input.keys, [block_c])


#TODO: implement cell response
class CellResponse(UnitResponse):

    def forward(self, 
                input: TensorMap, 
                systems: List[rascaline.torch.System]) -> TensorMap:

        outputs = list(torch.ones_like(input.block(0).values))

        dEdx = grad(outputs=list(input.block(0).values),
                      inputs=[sys_i.cell for sys_i in systems],
                      grad_outputs=outputs,
                      create_graph=True,
                      retain_graph=True)
        
        #print("hello")

        #negative forces, are position gradients
        gradient_values = torch.vstack(dEdx)

        cell_gradient_samples = Labels(
            ["sample", "structure", "cell"],
            torch.tensor(np.array(
                [
                    [s, s, a]
                    for s in range(len(systems))
                    for a in range(len(dEdx[s]))
                ]
            ))
        )