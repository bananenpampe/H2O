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
            values=gradient_values.reshape(-1, 3, 2),
            samples=position_gradient_samples,
            components=[Labels(["direction"], torch.arange(6).reshape(-1, 2))],
            properties=input.block(0).properties,
        )

        block_c = input.block(0).copy()
        block_c.add_gradient("positions", positions_gradient)

        #print(input.keys)

        return TensorMap(input.keys, [block_c])


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