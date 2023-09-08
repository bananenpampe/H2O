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

Begin with a fresh conda environment, (ie python 3.10) and activate the environment
```
conda create --name H2O python=3.10
```

Install rust, if it is not already installed:
```
conda install -c conda-forge rust
```

get rascaline + rascaline-torch + a cpu only torch version and metatensor, by running the three commands:

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu git+https://github.com/luthaf/rascaline#subdirectory=python/rascaline-torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-torch
pip install git+https://github.com/lab-cosmo/metatensor
```

finally install the requirements:

```
pip install -r requirements.txt
```




### Installing i-pi

Clone my i-pi fork:

```
git clone  https://github.com/bananenpampe/i-pi
```

change into the directory change the absolute path LIGHNING_CALCULATOR_PATH in `./i-pi/drivers/py/pes/lightning.py` to the absolute location of '/H2O/driver/'

```
from (in ./i-pi/drivers/py/pes/lightning.py)
LIGHNING_CALCULATOR_PATH = "/Users/matthiaskellner/Documents/PhD/H2O/driver/"
to
LIGHNING_CALCULATOR_PATH = "LOCATION_OF_H2O_REPOSITORY/H2O/driver/"

```

pip install in the root of the i-pi fork:

```
pip install .
```

finally change into the f90 part and make the fortran extensions

```
cd LOCATION_OF_IPI_REPOSITORY/i-pi/drivers/f90

make

```



## Train a model:

An example code for training with energies and forces
with a pytorch Lightning module, is provided in the example directory

For this simply run:

```

python example/training.py

```

## Run MD


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
