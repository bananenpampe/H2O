import torch
import equistore
import rascaline_torch
import numpy as np
from torch.autograd import grad
from typing import List

class UnitResponse(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # the unitresponse does nothing
    def initialize_weights(self, input: equistore.TensorMap):
        pass
    
    def forward(self, input: equistore.TensorMap, systems: List[rascaline_torch.System]) -> equistore.TensorMap:
        return input
        

class ForceRespone(UnitResponse):

    def forward(self, 
                input: equistore.TensorMap, 
                systems: List[rascaline_torch.System]) -> equistore.TensorMap:

        outputs = list(torch.ones_like(input.block(0).values))

        forces = grad(outputs=list(input.block(0).values),
                      inputs=[sys_i.positions for sys_i in systems],
                      grad_outputs=outputs,
                      create_graph=self.training)
        
        #print("hello")

        #negative forces, are position gradients
        gradient_values = -torch.vstack(forces)

        position_gradient_samples = equistore.Labels(
            ["sample", "structure", "atom"],
            np.array(
                [
                    [s, s, a]
                    for s in range(len(systems))
                    for a in range(len(forces[s]))
                ]
            ),
        )

        #TODO: change to torch labels once move to rascaline.rascaline-torch
        positions_gradient = equistore.TensorBlock(
            values=gradient_values.reshape(-1, 3, 1),
            samples=position_gradient_samples,
            components=[equistore.Labels(["direction"], np.arange(3).reshape(-1, 1))],
            properties=input.block(0).properties,
        )

        block_c = input.block(0).copy()
        block_c.add_gradient("positions", positions_gradient)

        #print(input.keys)

        return equistore.TensorMap(input.keys, [block_c])

