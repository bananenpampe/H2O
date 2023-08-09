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
from join import join

import torch 

def _disjoint_tensor_labels(tensors: List[TensorMap], axis: str) -> bool:
    """Checks if all labels in a list of TensorMaps are disjoint.

    We have to perform a check from all tensors to all others to ensure it
    they are "fully" disjoint.
    """
    for i_tensor, first_tensor in enumerate(tensors[:-1]):
        for second_tensor in tensors[i_tensor + 1 :]:
            for key, first_block in first_tensor.items():
                second_block = second_tensor.block(key)
                if axis == "samples":
                    first_labels = first_block.samples
                    second_labels = second_block.samples
                elif axis == "properties":
                    first_labels = first_block.properties
                    second_labels = second_block.properties

                if len(first_labels.intersection(second_labels)):
                    return False

    return True





def union_join(
    tensors: List[TensorMap], axis: str, remove_tensor_name: bool = False
) -> TensorMap:
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type of joining. For example, if
    ``axis='properties'`` the tensor maps in `tensors` will be joined along the
    `properties` dimension and for ``axis='samples'`` they will be the along the
    `samples` dimension.

    :param tensors:
        sequence of :py:class:`TensorMap` for join
    :param axis:
        A string indicating how the tensormaps are stacked. Allowed
        values are ``'properties'`` or ``'samples'``.
    :param remove_tensor_name:
        Remove the extra ``tensor`` dimension from labels if possible. See examples
        above for the case where this is applicable.

    :return tensor_joined:
        The stacked :py:class:`TensorMap` with more properties or samples
        than the input TensorMap.

    Examples
    --------
    Possible clashes of the meta data like ``samples``/``properties`` will be resolved
    by one of the three following strategies:

    1. If Labels names are the same, the values are unique and
       ``remove_tensor_name=True`` we keep the names and join the values

       >>> import numpy as np
       >>> import equistore
       >>> from equistore import Labels, TensorBlock, TensorMap

       >>> values = np.array([[1.1, 2.1, 3.1]])
       >>> samples = Labels("sample", np.array([[0]]))

       Define two disjoint :py:class:`Labels`.

       >>> properties_1 = Labels("n", np.array([[0], [2], [3]]))
       >>> properties_2 = Labels("n", np.array([[1], [4], [5]]))

       >>> block_1 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_1,
       ... )
       >>> block_2 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_2,
       ... )

       >>> tensor_1 = TensorMap(keys=Labels.single(), blocks=[block_1])
       >>> tensor_2 = TensorMap(keys=Labels.single(), blocks=[block_2])

       joining along the properties leads

       >>> joined_tensor = equistore.join(
       ...     [tensor_1, tensor_2], axis="properties", remove_tensor_name=True
       ... )
       >>> joined_tensor[0].properties
       Labels(
           n
           0
           2
           3
           1
           4
           5
       )

       If ``remove_tensor_name=False`` There will be an extra dimension ``tensor``
       added

       >>> joined_tensor = equistore.join(
       ...     [tensor_1, tensor_2], axis="properties", remove_tensor_name=False
       ... )
       >>> joined_tensor[0].properties
       Labels(
           tensor  n
             0     0
             0     2
             0     3
             1     1
             1     4
             1     5
       )

    2. If Labels names are the same but the values are not unique, a new dimension
       ``"tensor"`` is added to the names.

       >>> properties_3 = Labels("n", np.array([[0], [2], [3]]))

       ``properties_3`` has the same name and also shares values with ``properties_1``
       as defined above.

       >>> block_3 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_3,
       ... )
       >>> tensor_3 = TensorMap(keys=Labels.single(), blocks=[block_3])

       joining along properties leads to

       >>> joined_tensor = equistore.join([tensor_1, tensor_3], axis="properties")
       >>> joined_tensor[0].properties
       Labels(
           tensor  n
             0     0
             0     2
             0     3
             1     0
             1     2
             1     3
       )

    3. If Labels names are different we change the names to ("tensor", "property"). This
       case is only supposed to happen when joining in the property dimension, hence the
       choice of names:

       >>> properties_4 = Labels(["a", "b"], np.array([[0, 0], [1, 2], [1, 3]]))

       ``properties_4`` has the different names compared to ``properties_1``
       defined above.

       >>> block_4 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_4,
       ... )
       >>> tensor_4 = TensorMap(keys=Labels.single(), blocks=[block_4])

       joining along properties leads to

        >>> joined_tensor = equistore.join([tensor_1, tensor_4], axis="properties")
        >>> joined_tensor[0].properties
        Labels(
            tensor  property
              0        0
              0        1
              0        2
              1        0
              1        1
              1        2
        )
    """

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "the `TensorMap`s to join must be provided as a list or a tuple"
        )

    if len(tensors) < 1:
        raise ValueError("provide at least one `TensorMap` for joining")

    if axis not in ("samples", "properties"):
        raise ValueError(
            "Only `'properties'` or `'samples'` are "
            "valid values for the `axis` parameter."
        )

    if len(tensors) == 1:
        return tensors[0]

    for ts_to_join in tensors[1:]:
        _check_same_keys(tensors[0], ts_to_join, "join")
    
    all_keys = tensors[0].keys
    
    for tensor_i, tensor in enumerate(tensors):
        _, map, _ = all_keys.intersection_and_mapping(tensor.keys)

        missing_key_values = all_keys.values[map == -1]

        new_blocks = [block.copy() for block in tensor.blocks()]
        new_keys = Labels(
            names=tensor.keys.names,
            values=torch.vstack([tensor.keys.values, missing_key_values]),
        )
    
        for _ in missing_key_values:
            new_blocks.append(
                TensorBlock(
                    values=torch.empty([0, len(tensor[0].properties)]),
                    samples=Labels.empty(tensor.sample_names),
                    components=[],
                    properties=tensor[0].properties,
                )
            )


        tensors[tensor_i] = TensorMap(keys=new_keys, blocks=new_blocks)

    return join(tensors, axis=axis, remove_tensor_name=remove_tensor_name)