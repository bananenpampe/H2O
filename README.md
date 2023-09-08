# H2O
Various models learning the PES of H2O


## Get data

Execute the prepare.py script, which will download data from the AISsquare database:

```
python prepare.py
```

You can load the dataset via the utility functions:
```
from utils.load import load_PBE0_TS

# load the PBE0 TS dataset
ase_frames = load_PBE0_TS(which="lw_pmd")

# load the SCAN H2O phase dataset
ase_frames = load_phase_diagram_H2O()

```

## Install

get rascaline + rascaline-torch + a cpu only torch version

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu git+https://github.com/luthaf/rascaline#subdirectory=python/rascaline-torch
```

## Train a model:

An example code for training with energies and forces
with a pytorch Lightning module, is provided in the example directory

For this simply run:

```

python example/training.py

```


## Tests:

Modules should be tested for their symmetry, and size consistent / size extensive properties.
In case of the BPNN model, this will be outlined here:

- Test for size extensivity
- Test for size consitency (when considering average energies)
- Test for strict seperatability - having two non interacting systems
- Test for translational invariance
- Test for rotational invariance
- Test for invariance under listing-permutation of atoms

Since we initialize models that are not trained the following requirements have to be checked:
- predicted energies are nonzero
- predicted energies react to ratteling -> so we do not just add up the structure wise bias of the outlayers
