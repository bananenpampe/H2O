import ase
import numpy as np
import equistore

# get global number of species
def get_global_unique_species(frames):
    """returns a list of all unique species in the dataset
    """


    #TODO: test this
    # frame with one species
    # frame with multiple species
    # multiple frames with multiple species
    # multiple frames with one species
    # empty frame -> error

    if isinstance(frames, ase.Atoms):
        frames = [frames]

    species = []
    for frame in frames:
        species.extend(frame.get_atomic_numbers())

    return np.unique(species)


def _detach_all_blocks(tensor_map):
    blocks = []
    for _, block in tensor_map.items():
        new_block = equistore.TensorBlock(
            values=block.values.detach(),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)

            new_block.add_gradient(
                parameter=parameter,
                data=gradient.data.detach(),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)
    return equistore.TensorMap(tensor_map.keys, blocks)