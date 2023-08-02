import random
import numpy as np

import rascaline
import torch
from torch import autograd
import ase.io
import rascaline.torch

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = ase.io.read("../data/water_converted.xyz", index=":")
#print(torch.std(torch.tensor([float(len(frame)) for frame in frames_water])))
random.shuffle(frames_water)

frames_water = frames_water[:50]
frames_train = frames_water[:40]
frames_val = frames_water[40:]

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


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, frames, calc=rascaline.torch.SoapPowerSpectrum(**hypers_sr)):
        
        self.global_species = np.array([1,8])
        self.calc = calc

        self.energies = [torch.tensor(frame.info["TotEnergy"]) for frame in frames]
        self.forces = [torch.tensor(frame.arrays["force"]) for frame in frames]

        """self.frames = [rascaline_torch.as_torch_system(frame,
                                                       positions_requires_grad=True)
                                                         for frame in frames]"""
        
        self.frames = frames

        #self.feats = [self.calc(frame) for frame in self.frames]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.energies[idx], self.forces[idx],

def _collate_fn_w_custom_idx(batch):
    """Collate function for the dataloader.
    """

    print("compute_feats")
    systems = [rascaline.torch.system.systems_to_torch(item[0],positions_requires_grad=True)
                                                         for item in batch]

    calc_sr = rascaline.torch.SoapPowerSpectrum(**hypers_sr)

    feats = []
    for frame in systems:
        feat = calc_sr(frame)
        feat = feat.keys_to_samples("species_center")
        feat = feat.keys_to_properties(["species_neighbor_1","species_neighbor_2"])
        feat = feat.block(0).values
        feats.append(feat)
    
    feats = torch.vstack(feats)

    energies = torch.vstack([item[1] for item in batch])
    forces = torch.vstack([item[2] for item in batch])

    idx = torch.hstack([n*torch.ones_like(item[2][:,0],dtype=int) for n, item in enumerate(batch)])

    return feats, energies, forces, idx.flatten(), systems


dataset_train = SimpleDataset(frames_train)
dataset_val = SimpleDataset(frames_val)


dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=len(frames_train),
                                               shuffle=False,
                                               collate_fn=_collate_fn_w_custom_idx)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                               batch_size=len(frames_val),
                                               shuffle=False,
                                               collate_fn=_collate_fn_w_custom_idx)


print("create test loader")


# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

feats, energies, forces, idx, systems = next(iter(dataloader_train))


# ----- define a model -----

class ModelNoEqui(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in,25),
            torch.nn.Tanh(),
            torch.nn.Linear(25,25),
            torch.nn.Tanh(),
            torch.nn.Linear(25,n_out)
        )
    
    def forward(self, feat, idx):
        atomic_energies = self.model(feat)
        structure_energies = torch.zeros_like(torch.unique(idx).reshape(-1,1), dtype=torch.float64)
        structure_energies.index_add_(0, idx, atomic_energies)

        return structure_energies

mean_e = energies.mean()

# --- train the model ---

print("model init")
model = ModelNoEqui(feats.shape[1], 1)

n = 25
lambda_ = 1e-06
loss_lambda_ = 0.95

optimizer = torch.optim.LBFGS(model.parameters(), lr=1., line_search_fn="strong_wolfe")
loss_fn = torch.nn.MSELoss() #torch.nn.EnergyForceLoss(w_forces=False, force_weight=0.)
loss_fn_force = torch.nn.MSELoss()

for epoch in range(n):

    for (feats, energies_train, forces_train, idx, systems) in dataloader_train:

        model.train()

        energies_train = energies_train.clone() - mean_e

        steps = [0]


        def closure():

            optimizer.zero_grad()

            out = model.forward(feats, idx)

            dEdx = autograd.grad(outputs=out,
                        inputs=[sys_i.positions for sys_i in systems],
                        grad_outputs=torch.ones_like(out),
                        create_graph=True,
                        retain_graph=True)
            
            forces_pred = -torch.vstack(dEdx)

            loss = (1-loss_lambda_) * loss_fn(out.flatten(), energies_train.flatten())
            loss += loss_lambda_ * loss_fn_force(forces_pred.flatten(), forces_train.flatten())

            print("loss: ", float(loss))

            #do simple l2 regularization
            for param in model.parameters():
               loss += lambda_ * torch.norm(param)**2

            loss.backward(retain_graph=True)



            steps.append(1)
            return loss

        optimizer.step(closure)


        print("epoch: ", epoch)
        print("steps: ", sum(steps))


    for (feats, energies_val, forces_val, idx, systems) in dataloader_val:

        model.eval()

        out = model.forward(feats, idx)

        dEdx = autograd.grad(outputs=out,
                    inputs=[sys_i.positions for sys_i in systems],
                    grad_outputs=torch.ones_like(out),)
        

        forces_pred = -torch.vstack(dEdx)

        out += mean_e


        mse_energy = torch.nn.functional.mse_loss(energies_val.flatten(), out.flatten())
        mse_force = torch.nn.functional.mse_loss(forces_pred.flatten(),
                                                forces_val.flatten())

        print("mse_energy: ", float(mse_energy))
        print("mse_force: ",float(mse_force))
        print("%rmse energy: ", float(torch.sqrt(mse_energy))/float(energies_val.std()))
        print("%rmse force: ", float(torch.sqrt(mse_force))/float(forces_val.std()))