# initialize model from pytorch lightning .chkpt
# since we can get forces in forward mode, should we actually do it?
# how do we get the virial?
# write an additional response module that is force-cell ?

#dataset, how to set up calculators

import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import copy

# need to define hypers
# load example dataframe to initialize model

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "train"))

from nn.interaction import BPNNInteraction
from nn.model import BPNNModel
from rascaline_trainer_uncertainty import BPNNRascalineModule
from transformer.composition import CompositionTransformer
import torch
import rascaline
from nn.response import ForceUncertaintyRespone
from dataset.dataset import create_rascaline_dataloader, RascalineAtomisticDataset
import ase.io
from metatensor.torch import Labels
from torch.autograd import grad
import rascaline.torch
from rascaline.torch.utils import PowerSpectrum
import rascaline.torch

torch.set_default_dtype(torch.float64)

class PytorchLightningCalculator:

    def __init__(self, checkpoint, initial_frame):

        #uses initial frame to initialize model ?
        # should be changed to get a fully pickeled model, imo
        
        self.atoms = ase.io.read(initial_frame, index="0")
        
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

        """
        hypers_sr_spex = {
            "cutoff": 3.0,
            "max_radial": 4,
            "max_angular": 1,
            "atomic_gaussian_width": 0.3,
            "center_atom_weight": 1.0,
            "radial_basis": {
                "Gto": {},
            },
            "cutoff_function": {
                "ShiftedCosine": {"width":0.25},
            },
        }

        hypers_lr = {
            "cutoff": 3.0,
            "max_radial": 4,
            "max_angular": 1,
            "atomic_gaussian_width": 1.25,
            "center_atom_weight": 1.,
            "potential_exponent":1,
            "radial_basis": {
                "Gto": {},
            },
        }

        calc_sr_spex = rascaline.torch.SphericalExpansion(**hypers_sr_spex)
        calc_lr = rascaline.torch.LodeSphericalExpansion(**hypers_lr)

        calc_lode = PowerSpectrum(calc_sr_spex,calc_lr)
        """

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
        example_tensormap=feat,\
        model=BPNNModel(\
        interaction=BPNNInteraction(n_out=64, n_hidden_layers=2, activation=torch.nn.SiLU, n_hidden=64),
        response=ForceUncertaintyRespone()))
                
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
                                                         positions_requires_grad=True,)
                                                         #cell_requires_grad=True)
        
        feat_forward = self.dataset._compute_feats(forward_frame,
                                                   self.dataset.all_species)

        out = self.model.calculate(feat_forward, [forward_frame])
        
        # assumes constant number of molecules
        # take the inital frame and update positions and cell
        # wrap it as a pytorch, rascaline.torch system
        # on the dataset use _compute_feats() to get the features
        # call model.calculate() on the features
        # for now we return "useless" virial
        # should return energy, forces, virial

        energy = out.block(0).values
        forces = -out.block(0).gradient("positions").values

        energy = energy.detach().numpy()
        forces = forces.detach().numpy()

        """
        outputs = list(torch.ones_like(out.block(0).values))

        dEdc = grad(outputs=list(out.block(0).values),
                      inputs=[forward_frame.cell],
                      grad_outputs=outputs,
                      create_graph=True,
                      retain_graph=True)[0]
        
        #virial = -dEdc.detach().numpy().T @ cell_matrix
        virial = 0.5 * (virial + virial.T)
        """

        virial = cell_matrix * 0.0

        return energy, forces, virial 
