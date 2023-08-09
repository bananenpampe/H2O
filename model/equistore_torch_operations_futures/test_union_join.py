import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "utils"))

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer import BPNNRascalineModule
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
import rascaline
import rascaline.torch
import torch
import ase.io
import torch._dynamo
import traceback as tb
import random

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor

from equistore_torch_operations_futures.union_join import union_join
from equistore_torch_operations_futures.join import join

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = ase.io.read('../../data/nacl_train_sel.xyz', index=':')

frame_a  = frames_water[0]
frame_b = frames_water[100]

hypers_sr = {
    "cutoff": 5.6,
    "max_radial": 5,
    "max_angular": 5,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 2.0, "scale": 3.4}}
}




# --- define calulator ---
#calc_sr_ = rascaline.SoapPowerSpectrum(**hypers_sr)
calc_sr = rascaline.torch.SoapPowerSpectrum(**hypers_sr)

hypers_rs = {
    "cutoff": 5.6,
    "max_radial": 5,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 2.0, "scale": 3.4}}
}

calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)



dataloader_a = create_rascaline_dataloader([frame_a, frame_b],
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_sr],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=1, 
                                         shuffle=True)

loader_iterator = iter(dataloader_a)



feat_a, prop, syst = next(loader_iterator)


feat_b, prop, syst = next(loader_iterator)


for key, block in feat_a.items():
    print(key)
    print(block)
    print(block.values.shape)

for key, block in feat_b.items():
    print(key)
    print(block)
    print(block.values.shape)

#print(frame_a)
#print(frame_b)

#print(feat_a)
#print(feat_b)

out_map = union_join([feat_a, feat_b], axis="samples")

#print(out_map)

for key, block in out_map.items():
    print(key)
    print(block)
    print(block.values.shape)




"""
dataloader_a = create_rascaline_dataloader([frame_a, frame_b],
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=2, 
                                         shuffle=True,
                                         collate_fn=_equistore_collate)


"""

from collate_fn import union_collate_fn

dataloader_b = create_rascaline_dataloader([frame_a, frame_b],
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_sr, calc_rs],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=2, 
                                         shuffle=True,
                                         collate_fn=union_collate_fn)

x, prop, syst = next(iter(dataloader_b))

for key, block in x.items():
    print(key)
    print(block)
    print(block.values.shape)

print(x.block({"species_center":11}).values)

calculator = rascaline.SoapPowerSpectrum(**hypers_sr)
feat = calculator.compute([frame_a, frame_b])
#feat = feat.keys_to_samples("species_center")
feat = feat.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

"""
for key, block in feat.items():
    print(key)
    print(block)
    print(block.values.shape)
"""