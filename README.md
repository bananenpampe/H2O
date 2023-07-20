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

Train the model:

An example code for training with energies and forces
with a pytorch Lightning module, is provided in the example directory

For this simply run:

```

python example/training.py

```