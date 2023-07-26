import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "train"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "utils"))

from pytorch_lightning import Trainer
from rascaline_trainer import BPNNRascalineModule
from nn.model import BPNNModel
from nn.loss import EnergyForceLoss
from nn.response import UnitResponse
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
import rascaline
import torch
from transformer.composition import CompositionTransformer
import equistore
from pytorch_lightning.loggers import CSVLogger
from torch import autograd

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---
frames_water = load_PBE0_TS()
#print(torch.std(torch.tensor([float(len(frame)) for frame in frames_water])))
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

print("create train loader")

# --- create the dataloaders ---
dataloader_train = create_rascaline_dataloader(frames_train,
                                         calculators=calc_sr,
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train),
                                         shuffle=False)
print("create test loader")
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


mean_e = prop.block(0).values.mean()

# define the trainer
# module = BPNNRascalineModule(feat, transformer)#transformer)

# --- train the model ---

# create an empty text file and write losses to it later

print("model init")
model = BPNNModel(
    response = UnitResponse(),
)

model.initialize_weights(feat)

n = 10
lambda_ = 0.00
loss_lambda_ = 0.95

optimizer = torch.optim.LBFGS(model.parameters(), lr=1., line_search_fn="strong_wolfe")
loss_fn = torch.nn.MSELoss() #torch.nn.EnergyForceLoss(w_forces=False, force_weight=0.)
loss_fn_force = torch.nn.MSELoss()

for epoch in range(n):


    for (feat, prop, syst) in dataloader_train:

        #print(feat.block(0).values.requires_grad)
        #print(prop.block(0).values.requires_grad)
        #print(syst[0].positions.requires_grad)

        model.train()

        #with torch.no_grad():
        #    prop = transformer.transform(syst, prop)

        #print(prop.block(0).values.requires_grad)
        #print(syst[0].positions.requires_grad)

        prop_block = equistore.TensorBlock(values=(prop.block(0).values - mean_e),
                                                          samples = prop.block(0).samples,
                                                          components = prop.block(0).components,
                                                          properties = prop.block(0).properties)
        
        print(prop.block(0).samples)
        
        for parameter, gradient in prop.block(0).gradients():
            prop_block.add_gradient(parameter, gradient.copy())

        prop = equistore.TensorMap(prop.keys, [prop_block])


        #print(model.interaction.model.m_map["LabelsEntry(species_center=8)"].mean_out.weight)

        steps = [0]


        def closure():

            optimizer.zero_grad()
            #print("doing a step")
            out = model.forward(feat, syst)

            #we know that energies somehow -> learnable

            E = out.block(0).values

            dEdx = autograd.grad(outputs=list(E),
                        inputs=[sys_i.positions for sys_i in syst],
                        create_graph=True,
                        retain_graph=True)
            
            #print("hello")

            #negative forces, are position gradients
            forces = -torch.vstack(dEdx)


            loss = (1.-loss_lambda_) * loss_fn(E, prop.block(0).values)
            loss_force = loss_lambda_ * loss_fn_force(forces.flatten(),
                                                -prop.block(0).gradient("positions").values.flatten())


            #do simple l2 regularization
            for param in model.parameters():
               loss += lambda_ * torch.norm(param)**2

            loss.backward(retain_graph=True)

            steps.append(1)
            print("loss: ", float(loss_force))
            return loss

        optimizer.step(closure)

        print("epoch: ", epoch)
        print("steps: ", sum(steps))


    for (feat, prop, syst) in dataloader_val:

        model.eval()
        out = model.forward(feat, syst)
        #print(out.block(0).values)
        #print(transformer.transform(syst, prop).block(0).values)

        E = out.block(0).values

        dEdx = autograd.grad(outputs=list(E),
                    inputs=[sys_i.positions for sys_i in syst],
                    create_graph=True,
                    retain_graph=True)
        
        #print("hello")

        #negative forces, are position gradients
        forces = -torch.vstack(dEdx)

        #with torch.no_grad():
        #    out = transformer.inverse_transform(syst, out)
        mse_energy = torch.nn.functional.mse_loss(E + mean_e, prop.block(0).values)
        #print(prop.block(0).values.std())
        mse_force = torch.nn.functional.mse_loss(forces.flatten(),
                                  -prop.block(0).gradient("positions").values.flatten())

        print("mse_energy: ", float(mse_energy))
        print("mse_force: ",float(mse_force))
        print("%rmse energy: ", float(torch.sqrt(mse_energy))/float(prop.block(0).values.std()))
        print("%rmse force: ", float(torch.sqrt(mse_force))/float(prop.block(0).gradient("positions").values.std()))