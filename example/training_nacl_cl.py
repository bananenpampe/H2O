import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

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
from equistore_torch_operations_futures.collate_fn import union_collate_fn
import random

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor


#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = ase.io.read("../data/nacl_train_sel.xyz", index="2000:")
for frame in frames_water: frame.calc = None

"""
for frame in frames_water_:
    if 11 not in frame.numbers:
        continue
    else:
        frames_water.append(frame)
"""

for n, frame in enumerate(frames_water):
    frame.info["CONVERTED_ID"] = n

# shuffle the frames
SEED = 0
random.seed(SEED)
random.shuffle(frames_water)

# select a subset of the frames
frames_water_train = frames_water[:150]
frames_water_val = frames_water[150:175]
frames_water_test = frames_water[175:200]
frames_init = frames_water[200:500]

print(frames_water_train)

id_train = []
id_val = []
id_test = []

for frame in frames_water_train:
    id_train.append(frame.info["CONVERTED_ID"])

for frame in frames_water_val:
    id_val.append(frame.info["CONVERTED_ID"])

for frame in frames_water_test:
    id_test.append(frame.info["CONVERTED_ID"])

# --- define the hypers ---
hypers_sr = {
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
calc_sr = rascaline.torch.SoapPowerSpectrum(**hypers_sr)
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)

# --- create the dataloader ---
dataloader_init = create_rascaline_dataloader(frames_init,
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_rs],
                                         do_gradients= False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = union_collate_fn,
                                         batch_size=len(frames_init), 
                                         shuffle=False)

dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_sr],
                                         do_gradients= True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = union_collate_fn,
                                         batch_size=4, 
                                         shuffle=True)

dataloader_val = create_rascaline_dataloader(frames_water_val,
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_sr],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = union_collate_fn,
                                         batch_size=len(frames_water_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_water_test,
                                         energy_key="potential",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_sr],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         collate_fn = union_collate_fn,
                                         batch_size=len(frames_water_test), 
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
wandb_logger = WandbLogger(project="H2O-NaCl",log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams(hypers_sr)

print("train split:",id_train)
print("test_split:",id_test)
print("seed", SEED)



"""
for batch in dataloader:
    feat, prop, syst = batch

    print("----------- batch start ----------")
    for key, block in feat.items():
        print(key)
        print(block)
        print(block.values.shape)
    
    print("----------- batch end ----------")
"""

_, prop, syst = next(iter(dataloader_init))

transformer_e = CompositionTransformer()
transformer_e.fit(syst, prop)

feat, prop, syst = next(iter(dataloader))
# define the trainer
module = BPNNRascalineModule(feat, transformer_e)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=100,
                  precision=64,
                  accelerator="cpu",
                  logger=wandb_logger,
                  callbacks=[lr_monitor],
                  gradient_clip_val=100,
                  enable_progress_bar=False,
                  val_check_interval=1.0,
                  check_val_every_n_epoch=1,
                  inference_mode=False)
                  #profiler="simple")

trainer.fit(module, dataloader, dataloader_val)
trainer.test(module, dataloaders=dataloader_test)