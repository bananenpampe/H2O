import torch
import torch
import copy
import metatensor
from metatensor.torch import sum_over_samples
from .feature import UnitFeatures
from .interaction import BPNNInteraction
from .aggregation import StructureWiseAggregation, BPNNStructureWiseAggregation
from .response import UnitResponse, ForceRespone


class DeepEnsemble(torch.nn.Module):

    """Constructs a Deep Ensemble from N*Mean_Var MLPs
    """
    
    def __init__(self, ensembles, kind) -> None:
        super().__init__()

        assert kind in ["deep-ens","mse-deep-ens"]
        
        #TODO: changed from older notebook
        
        self.ensembles = torch.nn.ModuleList(ensembles)
        self.kind = kind

    def forward(self, inputs, systems):
        
        out_pred = []

        for ens in self.ensembles:
            
            out = ens.calculate(inputs, systems)
            out_pred.append(out)

        return out_pred
    
    def report_energy_forces(self, inputs, systems, report_aleatoric=False):

        if report_aleatoric:
            assert self.kind == "deep-ens"
        
        out_pred = self.forward(inputs, systems)
        E_pred = torch.stack([out.block(0).values for out in out_pred], dim=1)
        F_pred = torch.stack([out.block(0).gradient("positions").values for out in out_pred], dim=1)
        F_pred_mean = torch.mean(F_pred, dim=1)

        E_UQ_epistemic = torch.var(E_pred, dim=1)
        F_UQ_epistemic = torch.var(F_pred, dim=1)

        E_pred_mean = torch.mean(E_pred, dim=1)

        if self.kind == "deep-ens":
            E_pred_var = torch.stack([out.block(1).values for out in out_pred], dim=1)

            #currently force UQ not implemented
            #F_pred_var = torch.stack([out.block(0).gradient("positions") for out in out_pred], dim=1)
            
            E_UQ_aleatoric = torch.mean(E_pred_var, dim=1)
            E_UQ = E_UQ_aleatoric + E_UQ_epistemic
        
        elif self.kind == "mse-deep-ens":
            E_UQ = E_UQ_epistemic
        else:
            raise NotImplementedError
        
        if report_aleatoric:
            return E_pred_mean, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred_mean, F_UQ_epistemic,
        else:
            return E_pred_mean, E_UQ, F_pred_mean, F_UQ_epistemic
        