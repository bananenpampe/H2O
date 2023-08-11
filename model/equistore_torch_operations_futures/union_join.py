import functools
import operator
from typing import List

import numpy as np

from equistore.torch import Labels, TensorBlock, TensorMap
import equistore.torch

from empty_like import empty_like_block
from join import join

from equistore.operations._utils import _check_same_keys
from equistore.operations.manipulate_dimension import remove_dimension

import torch 

def union_join(tensors: List[TensorMap], axis: str):
    dtype_ = tensors[0].block(0).values.dtype
    # Constrauct a Labels object with all keys
    all_keys = tensors[0].keys
    for tensor in tensors[1:]:
        all_keys = all_keys.union(tensor.keys)

    # Create empty blocks for missing keys for each TensorMap
    new_tensors = []
    for tensor in tensors:
        _, map, _ = all_keys.intersection_and_mapping(tensor.keys)

        missing_keys = Labels(
            names=tensor.keys.names, values=all_keys.values[map == -1]
        )

        new_keys = tensor.keys.union(missing_keys)
        new_blocks = [block.copy() for block in tensor.blocks()]

        for key in missing_keys:
            # Find corresponding block with the missing key
            reference_tensor = None
            for reference_tensor in tensors:
                if key in reference_tensor.keys:
                    reference_block = reference_tensor.block(key)
                    break

            # There should be a block with the key otherwise we did something wrong
            assert reference_tensor is not None

            # Construct new block with 0 samples based on the metadata of reference_block

            new_block = TensorBlock(
                values=torch.empty((0,) + reference_block.values.shape[1:], dtype=dtype_),
                samples=Labels.empty(reference_block.samples.names),
                components=reference_block.components,
                properties=reference_block.properties,
            )

            for parameter, gradient in reference_block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=torch.empty((0,) + gradient.values.shape[1:], dtype=dtype_),
                        samples=Labels.empty(gradient.samples.names),
                        components=gradient.components,
                        properties=new_block.properties,
                    ),
                )

            new_blocks.append(new_block)

        new_tensors.append(TensorMap(keys=new_keys, blocks=new_blocks))

    return join(new_tensors, axis=axis)