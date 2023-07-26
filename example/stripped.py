import random
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "utils"))

from pytorch_lightning import Trainer
from rascaline_trainer import BPNNRascalineModule
from nn.model import BPNNModel
from nn.loss import EnergyForceLoss
from nn.response import UnitResponse
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader, _equistore_collate_w_custom_idx
import rascaline
import torch
from transformer.composition import CompositionTransformer
import equistore
from pytorch_lightning.loggers import CSVLogger
from torch import autograd
import ase.io

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = ase.io.read("../data/water_converted.xyz", index=":")
#print(torch.std(torch.tensor([float(len(frame)) for frame in frames_water])))
random.shuffle(frames_water)

frames_water = frames_water[:20]
frames_train = frames_water[:10]
frames_val = frames_water[10:]

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

"""
dataloader_setup = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,#setup,
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=False)

transformer = CompositionTransformer()
feat, prop, syst = next(iter(dataloader_setup))
prop = equistore.to(prop,"torch",dtype=torch.float64)
"""

"""

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, frames, calc=rascaline.SoapPowerSpectrum(**hypers_sr)):
        
        self.global_species = np.array([1,8])
        self.frames = frames 
        self.calc = calc




    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.calc(self.frames[idx])

"""

print("create train loader")

# --- create the dataloaders ---
dataloader_train = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         energy_key="TotEnergy",
                                         forces_key="force",
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train),
                                         shuffle=False,
                                         collate_fn=_equistore_collate_w_custom_idx)

print("create test loader")
dataloader_val = create_rascaline_dataloader(frames_val,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute =  True,
                                         lazy_fill_up = False,
                                         energy_key="TotEnergy",
                                         forces_key="force",
                                         batch_size=len(frames_val),
                                         shuffle=False,
                                         collate_fn=_equistore_collate_w_custom_idx)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

feats, energies, forces, idx, systems = next(iter(dataloader_train))


# ----- define a model -----

class ModelNoEqui(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,n_out)
        )
    
    def forward(self, feat, idx):
        atomic_energies = self.model(feat)
        structure_energies = torch.zeros_like(torch.unique(idx).reshape(-1,1), dtype=torch.float64)
        structure_energies.index_add_(0, idx, atomic_energies)

        return structure_energies

mean_e = energies.mean()

# define the trainer
# module = BPNNRascalineModule(feat, transformer)#transformer)

# --- train the model ---

# create an empty text file and write losses to it later

print("model init")
model = ModelNoEqui(feats.shape[1], 1)

n = 25
lambda_ = 0.01
loss_lambda_ = 0.95

optimizer = torch.optim.LBFGS(model.parameters(), lr=1., line_search_fn="strong_wolfe")
loss_fn = torch.nn.MSELoss() #torch.nn.EnergyForceLoss(w_forces=False, force_weight=0.)
loss_fn_force = torch.nn.MSELoss()

for epoch in range(n):


    for (feats, energies_train, forces_train, idx, systems) in dataloader_train:

        #print(feat.block(0).values.requires_grad)
        #print(prop.block(0).values.requires_grad)
        #print(syst[0].positions.requires_grad)

        model.train()

        #with torch.no_grad():
        #    prop = transformer.transform(syst, prop)

        #print(prop.block(0).values.requires_grad)
        #print(syst[0].positions.requires_grad)

        #print(energies_train)

        energies_train -= mean_e

        #print(energies_train)


        #print(model.interaction.model.m_map["LabelsEntry(species_center=8)"].mean_out.weight)

        steps = [0]


        def closure():

            optimizer.zero_grad()
            #print("doing a step")
            out = model.forward(feats, idx)

            #we know that energies somehow -> learnable

            #dEdx = autograd.grad(outputs=list(out),
            #            inputs=[sys_i.positions for sys_i in systems],
            #            create_graph=True,
            #            retain_graph=True)

            dEdx = autograd.grad(outputs=out,
                        inputs=[sys_i.positions for sys_i in systems],
                        grad_outputs=torch.ones_like(out),
                        create_graph=True,
                        retain_graph=True)
            
            #print("hello")

            #negative forces, are position gradients
            forces_pred = -torch.vstack(dEdx)


            loss = loss_fn(out.flatten(), energies_train.flatten())
            loss += loss_fn_force(forces_pred.flatten(), forces_train.flatten())

            print("loss: ", float(loss))

            #do simple l2 regularization
            for param in model.parameters():
               loss += lambda_ * torch.norm(param)**2

            loss.backward(retain_graph=True)
            steps.append(1)
            #print("loss: ", float(loss_force))
            return loss

        optimizer.step(closure)

        print("epoch: ", epoch)
        print("steps: ", sum(steps))


    for (feats, energies_val, forces_val, idx, systems) in dataloader_val:

        model.eval()
        out = model.forward(feats, idx)
        #print(out.block(0).values)
        #print(transformer.transform(syst, prop).block(0).values)

        dEdx = autograd.grad(outputs=out,
                    inputs=[sys_i.positions for sys_i in systems],
                    grad_outputs=torch.ones_like(out),
                    create_graph=True,
                    retain_graph=True)
        
        #print("hello")

        #negative forces, are position gradients
        forces_pred = -torch.vstack(dEdx)

        out += mean_e

        #with torch.no_grad():
        #    out = transformer.inverse_transform(syst, out)
        mse_energy = torch.nn.functional.mse_loss(energies_val.flatten(), out.flatten())
        #print(prop.block(0).values.std())
        mse_force = torch.nn.functional.mse_loss(forces_pred.flatten(),
                                                 forces_val.flatten())

        print("mse_energy: ", float(mse_energy))
        print("mse_force: ",float(mse_force))
        print("%rmse energy: ", float(torch.sqrt(mse_energy))/float(energies_val.std()))
        print("%rmse force: ", float(torch.sqrt(mse_force))/float(forces_val.std()))