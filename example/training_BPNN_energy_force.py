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
from transformer.composition import CompositionTransformer
import equistore

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = load_PBE0_TS()[:150]
frames_train = frames_water[:100]
frames_val = frames_water[100:150]
frames_test = frames_water[20:30]

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

hypers_setup = {
    "cutoff": 3.0,
    "max_radial": 2,
    "max_angular": 2,
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
calc_setup = rascaline.SoapPowerSpectrum(**hypers_setup)


dataloader_setup = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,#setup,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=False)

transformer = CompositionTransformer()
feat, prop, syst = next(iter(dataloader_setup))
prop = equistore.to(prop,"torch",dtype=torch.float64)
transformer.fit(syst, prop)

del feat, prop, syst
del dataloader_setup
del calc_setup


# --- create the dataloaders ---
dataloader_train = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=2, 
                                         shuffle=False)

dataloader_val = create_rascaline_dataloader(frames_val,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute =  True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_test,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_test), 
                                         shuffle=False)


# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

feat, prop, syst = next(iter(dataloader_train))

# define the trainer
module = BPNNRascalineModule(feat, transformer)#transformer)

# --- train the model ---
n = 5

trainer = Trainer(max_epochs=n, precision=64, accelerator="cpu", inference_mode=False,num_sanity_val_steps=0)

# create an empty text file and write losses to it later

with open("losses.txt", "w") as f:
    # write a losses header to file
    f.write("epoch, energy_mse, force_mse")



for i in range(25):

    print("epoch: ", i)
    trainer.fit(module, dataloader_train)

    with torch.enable_grad():
        feat, prop, syst = next(iter(dataloader_val))
        out = module.forward(feat, syst)
        out = out.copy()

    with torch.no_grad():
        out = transformer.inverse_transform(syst, out)
        energy_mse, force_mse = module.loss_fn.report(out, prop)

    print("energy_mse: ", energy_mse)
    print("force_mse: ", force_mse)

    #write losses to file, scv style
    with open("losses.txt", "a") as f:
        f.write("\n")
        f.write(str(i) + ", " + str(energy_mse) + ", " + str(force_mse))


    trainer = Trainer(max_epochs=n, precision=64, accelerator="cpu", inference_mode=False, num_sanity_val_steps=0)


#trainer.validate(module, dataloader_val)

#trainer.validate(module)
#trainer.test(dataloader_test)
