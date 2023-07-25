import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

from pytorch_lightning import Trainer
from rascaline_trainer import BPNNRascalineModule
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
import rascaline
import torch

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = load_PBE0_TS()[:100]

# --- define the hypers ---
hypers_sr = {
    "cutoff": 3.0,
    "max_radial": 5,
    "max_angular": 3,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 3.0, "rate": 1.5, "scale": 2.0}}
}

# --- define calculator ---
calc_sr = rascaline.SoapPowerSpectrum(**hypers_sr)

# --- create the dataloader ---
dataloader = create_rascaline_dataloader(frames_water,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=10, 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

feat, prop, syst = next(iter(dataloader))

# define the trainer
module = BPNNRascalineModule(feat)

# --- train the model ---
trainer = Trainer(max_epochs=5, precision=64)
trainer.fit(module, dataloader)