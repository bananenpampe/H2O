# %%
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.join("..", "model"))
sys.path.append(os.path.join("..", "train"))
sys.path.append(os.path.join("..",  "utils"))

from nn.response import   ForceUncertaintyRespone
from nn.model import BPNNModel
from nn.interaction import BPNNInteraction
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer_uncertainty_shieldings import BPNNRascalineModule
from load import load_PBE0_TS, load_phase_diagram_H2O
from dataset.dataset import create_rascaline_dataloader
import rascaline
import rascaline.torch
import torch
import ase.io
import torch._dynamo
import traceback as tb
from metatensor_torch_operations_futures.collate_fn import union_collate_fn, metatensor_collate_sort
import random

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor

from equisolve_futures.convert_torch import ase_scalar_to_tensormap
import random

SEED = 0
random.seed(SEED)


#default type is float64
torch.set_default_dtype(torch.float64)


# %%
frames_raw = ase.io.read("../../NMR/CSD-3k+S546_shift_tensors_w_xtb_feat_GFN1_complete.xyz", ":")
frames_filt = []

for frame in frames_raw:
    if frame.info["STATUS"] == "PASSING":
        frames_filt.append(frame)

#del frames_raw
random.shuffle(frames_filt)

frames_train = frames_filt[:3300][:200]
frames_val = frames_filt[3300:][:50]


frames_test = ase.io.read("../../NMR/CSD-500+104-7_shift_tensors_w_xtb_feat_GFN1_complete.xyz", ":")
random.shuffle(frames_test)
frames_test = frames_test[:50]

hypers_ps = {
    "cutoff": 5.,
    "max_radial": 3,
    "max_angular": 3,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 2.0, "scale": 2.0}}
}

hypers_rs = {
    "cutoff": 5.,
    "max_radial": 16,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 2.0, "scale": 2.0}}
}


# --- define calculator ---
#calc_sr_ = rascaline.SoapPowerSpectrum(**hypers_sr)
calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)

# %%
dataloader_train = create_rascaline_dataloader(frames_train,
                                         energy_key="cs_iso",                                      
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients= False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = metatensor_collate_sort,
                                         batch_size=4, 
                                         shuffle=False,
                                         atomistic=True,
                                         filter_by_central_id=6)

dataloader_val = create_rascaline_dataloader(frames_val,
                                         energy_key="cs_iso",                                      
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients= True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = metatensor_collate_sort,
                                         batch_size=len(frames_val), 
                                         shuffle=False,
                                         atomistic=True,
                                         filter_by_central_id=6)

dataloader_test = create_rascaline_dataloader(frames_test,
                                         energy_key="cs_iso",                                      
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients= True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = metatensor_collate_sort,
                                         batch_size=len(frames_test), 
                                         shuffle=False,
                                         atomistic=True,
                                         filter_by_central_id=6)



# %%
with open("wandb.txt", "r") as f:
    #read the first line
    wandb_api_key = f.readline()

wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(project="learn-CS",log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams({"hypers radial spectrum": hypers_rs})
wandb_logger.log_hyperparams({"hypers power spectrum": hypers_ps})

print("seed", SEED)



all_spec = dataloader_train.dataset.all_species

expected_keys = []
for spec in all_spec:
    expected_keys.append("LabelsEntry(species_center={})".format(spec))

expected_keys = set(expected_keys)

for n, (feat, _, _) in enumerate(dataloader_train):
    tmp_keys = []
    for key, t_block in feat.items():
        tmp_keys.append(str(key))    
    if set(tmp_keys) == expected_keys:
        NO_FULL = False
        break 

module = BPNNRascalineModule(feat)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=300,
                  precision=64,
                  accelerator="cpu",
                  logger=wandb_logger,
                  callbacks=[lr_monitor],
                  gradient_clip_val=0.5,
                  enable_progress_bar=False,
                  val_check_interval=1.0,
                  check_val_every_n_epoch=1,
                  inference_mode=False)
                  #profiler="simple")

trainer.fit(module, dataloader_train, dataloader_val)
trainer.test(module, dataloader_test)
