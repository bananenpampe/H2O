import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

from pytorch_lightning import Trainer
from rascaline_trainer import BPNNRascalineModule
from nn.model import BPNNModel
from nn.loss import EnergyForceLoss
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
import rascaline
import torch
from transformer.composition import CompositionTransformer
import equistore
from pytorch_lightning.loggers import CSVLogger

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = load_PBE0_TS()
random.shuffle(frames_water)

frames_water = frames_water[:10]
frames_train = frames_water[:5]
frames_val = frames_water[5:]

# --- define the hypers ---
hypers_sr = {
    "cutoff": 5.0,
    "max_radial": 5,
    "max_angular": 5,
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


# --- create the dataloaders ---
dataloader_train = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=False)

dataloader_val = create_rascaline_dataloader(frames_val,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute =  True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_val), 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

feat, prop, syst = next(iter(dataloader_train))

# define the trainer
# module = BPNNRascalineModule(feat, transformer)#transformer)

# --- train the model ---

# create an empty text file and write losses to it later

model = BPNNModel()
model.initialize_weights(feat)

n = 10

optimizer = torch.optim.LBFGS(model.parameters(), lr=1.)
loss_fn = EnergyForceLoss(w_forces=True, force_weight=0.95)

for epoch in range(n):
    
    
    for (feat, prop, syst) in dataloader_train:
        
        model.train()
        
        prop = transformer.transform(syst, prop)

        def closure():
            
            optimizer.zero_grad()
            out = model.forward(feat, syst)
            loss = loss_fn(out, prop)
            loss.backward(retain_graph=True)
            
            return loss
        
        optimizer.step(closure)

    
    for (feat, prop, syst) in dataloader_val:

        model.eval()

        out = model.forward(feat, syst)

        print(out.block(0).values)
        print(transformer.transform(syst, prop).block(0).values)

        out = transformer.inverse_transform(syst, out)

        mse_energy = torch.nn.functional.mse_loss(out.block(0).values, prop.block(0).values)
        mse_force = torch.nn.functional.mse_loss(out.block(0).gradient("positions").values,
                                                 prop.block(0).gradient("positions").values)
        
        print(prop.block(0).values.std())
        
        print("mse_energy: ", float(mse_energy))
        print("mse_force: ",float(mse_force))
        print("%rmse energy: ", float(torch.sqrt(mse_energy))/float(prop.block(0).values.std()))
        print("%rmse force: ", float(torch.sqrt(mse_energy))/float(prop.block(0).gradient("positions").values.std()))


