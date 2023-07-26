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
