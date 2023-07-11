# H2O
Various models learning the PES of H2O


## Get data

Execute the prepare.py script in the data directory:
```
cd data
python prepare.py
```

You can load the dataset via the utility functions:
```
from utils.load import load_PBE0_TS

ase_frames = load_PBE0_TS(which="lw_pmd")
```

