import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer_uncertainty_QM9 import BPNNRascalineModule
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
from dataset.dataset_helpers import get_global_unique_species
import rascaline
import rascaline.torch
import torch
import ase.io
import torch._dynamo
import traceback as tb
import random

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor


#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_qm9 = ase.io.read("../data/qm9.xyz", index=":")
frames_filtered = []
for frame in frames_qm9:
    if frame.info['problematic'] == "OK":
        frames_filtered.append(frame)

# shuffle the frames
SEED = 0
random.seed(SEED)
random.shuffle(frames_filtered)

for n, frame in enumerate(frames_filtered):
    frame.info["CONVERTED_ID"] = n

# select a subset of the frames
frames_train = frames_filtered[:1000]
frames_val = frames_filtered[1000:1100]
frames_test = frames_filtered[1200:1300]

id_train = []
id_val = []
id_test = []

for frame in frames_train:
    id_train.append(frame.info["CONVERTED_ID"])

for frame in frames_val:
    id_val.append(frame.info["CONVERTED_ID"])

for frame in frames_test:
    id_test.append(frame.info["CONVERTED_ID"])

# --- define the hypers ---
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
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)
calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)

# --- create the dataloader ---
dataloader_init = create_rascaline_dataloader(frames_train,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=True)

dataloader = create_rascaline_dataloader(frames_train,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=32, 
                                         shuffle=True)

dataloader_val = create_rascaline_dataloader(frames_val,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_test,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_test), 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

#COPY YOUR WANDB API KEY HERE, or load it fromn a file

#read wandb api code from file
with open("wandb.txt", "r") as f:
    #read the first line
    wandb_api_key = f.readline()

wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(project="learn-QM9",log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams({"hypers radial spectrum": hypers_rs})
wandb_logger.log_hyperparams({"hypers power spectrum": hypers_ps})

print("train split:",id_train)
print("test_split:",id_test)
print("seed", SEED)

feat, prop, syst = next(iter(dataloader_init))

transformer_e = CompositionTransformer()
transformer_e.fit(syst, prop)





all_spec = dataloader_init.dataset.all_species

expected_keys = []
for spec in all_spec:
    expected_keys.append("LabelsEntry(species_center={})".format(spec))

expected_keys = set(expected_keys)

for n, (feat, _, _) in enumerate(dataloader):
    tmp_keys = []
    for key, t_block in feat.items():
        tmp_keys.append(str(key))    
    if set(tmp_keys) == expected_keys:
        NO_FULL = False
        break 

# define the trainer
module = BPNNRascalineModule(feat, transformer_e)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=5,
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

trainer.fit(module, dataloader, dataloader_val)
#trainer.test(module, dataloaders=dataloader_test)