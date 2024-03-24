# initialize model from pytorch lightning .chkpt
# since we can get forces in forward mode, should we actually do it?
# how do we get the virial?
# write an additional response module that is force-cell ?

#dataset, how to set up calculators

import pytorch_lightning as pl
import copy

# need to define hypers
# load example dataframe to initialize model

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))

from nn.interaction import BPNNInteraction
from nn.response import ForceRespone
from nn.model import BPNNModel
from rascaline_trainer import BPNNRascalineModule
from transformer.composition import CompositionTransformer
import torch
import rascaline
from nn.response import ForceUncertaintyRespone
from dataset.dataset import RascalineAtomisticDataset
import ase.io
from metatensor.torch import Labels
from torch.autograd import grad
import rascaline.torch
import rascaline.torch
import json
import numpy as np
from ipi.utils.units import unit_to_internal

torch.set_default_dtype(torch.float64)

class PytorchLightningCalculator:

    def __init__(self, checkpoint, initial_frame):
        
        self.atoms = ase.io.read(initial_frame, index="0")

        self.atoms.info["potential"] = 0.0
        self.atoms.arrays["force"] = np.ones_like(self.atoms.get_positions())
        
        checkpoint = torch.load(checkpoint)['state_dict']

        hypers_ps = {
            "cutoff": 5.,
            "max_radial": 5,
            "max_angular": 5,
            "atomic_gaussian_width": 0.25,
            "center_atom_weight": 0.0,
            "radial_basis": {
                "Gto": {},
            },
            "cutoff_function": {
                "ShiftedCosine": {"width":0.5},
            },
            "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 3.0, "scale": 2.0}}
        }

        hypers_rs = {
            "cutoff": 5.,
            "max_radial": 16,
            "atomic_gaussian_width": 0.25,
            "center_atom_weight": 0.0,
            "radial_basis": {
                "Gto": {},
            },
            "cutoff_function": {
                "ShiftedCosine": {"width":0.5},
            },
            "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 3.0, "scale": 2.0}}
        }

        calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)
        calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)

        self.dataset = RascalineAtomisticDataset(self.atoms,
                                         energy_key="potential",
                                         forces_key="force",
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False)
        

        # ----- initialize model weights with features -----
        ex_frame = rascaline.torch.systems_to_torch(copy.deepcopy(self.atoms))
        feat = self.dataset._compute_feats(ex_frame, self.dataset.all_species)

        self.model = BPNNRascalineModule(\
        energy_transformer=CompositionTransformer(multi_block=True),
        example_tensormap=feat,\
        model=BPNNModel(\
        interaction=BPNNInteraction(n_out=5, n_hidden_layers=2, activation=torch.nn.SiLU, n_hidden=64),
        response=ForceUncertaintyRespone()),)
                
        print(self.model)
        print(self.model.state_dict().keys())
        # ----- load model from checkpoint -----
        
        if "energy_transformer.weights" in checkpoint.keys():
            print("found transformer weights")
        else:
            print("setting transformer weights to 0.")
            
            checkpoint["energy_transformer.weights"] = torch.nn.Parameter(torch.tensor([0. for i in self.dataset.all_species]).reshape(-1,1),
                                                                    requires_grad=False)
             
        self.model.energy_transformer.weights = checkpoint["energy_transformer.weights"]
        checkpoint.pop("energy_transformer.weights")
        self.model.load_state_dict(checkpoint)
        self.model.energy_transformer.is_fitted = True
        self.model.energy_transformer.unique_labels = Labels(["species_center"], values=torch.tensor(self.dataset.all_species).reshape(-1,1))
        print(self.model)



    def calculate(self, positions, cell_matrix):

        self.atoms.set_positions(positions)
        self.atoms.set_cell(cell_matrix)

        forward_frame = rascaline.torch.systems_to_torch(self.atoms,
                                                         positions_requires_grad=True,
                                                         cell_requires_grad=True)
        
        feat_forward = self.dataset._compute_feats(forward_frame,
                                                   self.dataset.all_species)

        out = self.model.calculate(feat_forward, [forward_frame])
        
        energy = out.block(0).values
        force = -out.block(0).gradient("positions").values

        outputs = list(torch.ones_like(out.block(0).values))

        dEdc = grad(outputs=list(out.block(0).values),
                      inputs=[forward_frame.cell],
                      grad_outputs=outputs,
                      create_graph=True,
                      retain_graph=True)[0]
        
        virial = -dEdc.detach().numpy().T @ cell_matrix
        virial = 0.5 * (virial + virial.T)

        energy = energy.detach().numpy()
        force = force.detach().numpy()

        committee_e = self.model.model.get_energy(
            feat_forward, [forward_frame]
            ).block(0).values

        composition_e = self.model.energy_transformer.get_composition_energy([forward_frame])

        forces = []

        force_for_dyn = []
        virials = []
            
        for e_i in committee_e.flatten():
            
            # Zero-out previous gradients if any
            
            if forward_frame.positions.grad is not None:
                forward_frame.positions.grad.zero_()
            
            if forward_frame.cell.grad is not None:
                forward_frame.cell.grad.zero_()

            e_i.backward(retain_graph=True)

            f_np = -forward_frame.positions.grad.clone().numpy()
            f_np = unit_to_internal("force", "ev/ang", f_np)

            forces.append(f_np.flatten()) 

            v_np = -forward_frame.cell.grad.detach().numpy().T @ cell_matrix
            v_np = 0.5 * (v_np + v_np.T)

            v_np = unit_to_internal("energy", "electronvolt", v_np)

            virials.append(v_np.flatten()) 
        

        committee_e += composition_e
        committee_e = committee_e.detach().numpy().reshape(-1,1)

        
        forces = np.stack(forces)
        virials = np.stack(virials)

        #write to std out the shape of forces and virials and committee_e using sys.stdout.write
        #print(forces.shape, virials.shape, committee_e.shape)

        committee_e = unit_to_internal("energy", "electronvolt", committee_e) 
        

        extras = {"committee_pot":committee_e.tolist(),
                  "committee_force":forces.tolist(),
                  "committee_virial":virials.tolist()}
        
        extras = json.dumps(extras)

        return energy, force, virial, extras