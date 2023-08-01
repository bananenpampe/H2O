import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))

from nn.model import BPNNModel
from nn.loss import EnergyForceLoss

class BPNNRascalineModule(pl.LightningModule):
    
    def __init__(self, example_tensormap, energy_transformer, regularization=1e-03):
        super().__init__()
        #print(regularization)
        #self.save_hyperparameters({'l2 reg': regularization})
        self.model = BPNNModel()
        self.model.initialize_weights(example_tensormap)
        self.loss_fn = EnergyForceLoss(w_forces=True, force_weight=0.95)
        self.energy_transformer = energy_transformer
        self.regularization = regularization
        self.automatic_optimization = False

    def forward(self, feats, systems):
        return self.model(feats, systems)
    

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
       
        def closure():
            feats, properties, systems = batch
            
            batch_size = len(systems)
            properties = self.energy_transformer.transform(systems, properties)

            outputs = self(feats,systems)
            loss = self.loss_fn(outputs, properties)

            self.log('train_loss', loss, enable_graph=True, batch_size=batch_size)

            for param in self.parameters():
                loss += self.regularization * torch.norm(param)**2
            
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        opt.step(closure=closure)

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):

        feats, properties, systems = batch

        energies = properties.block(0).values 
        forces = properties.block(0).gradient("positions").values

        batch_size = len(systems)

        outputs = self(feats, systems)
        outputs = self.energy_transformer.inverse_transform(systems, outputs)

        energy_val_mse, forces_val_mse = self.loss_fn.report(outputs, properties)

        loss = energy_val_mse + forces_val_mse

        self.log('val_loss', loss.item(), batch_size=batch_size)
        self.log("val_energy_mse", torch.clone(energy_val_mse), batch_size=batch_size)
        self.log("val_forces_mse", torch.clone(forces_val_mse), batch_size=batch_size)

        # log rmse
        self.log("val_energy_rmse", torch.sqrt(torch.clone(energy_val_mse)), batch_size=batch_size)
        self.log("val_forces_rmse", torch.sqrt(torch.clone(forces_val_mse)), batch_size=batch_size)

        # log percent rmse
        self.log("val_energy_%rmse", torch.sqrt(torch.clone(energy_val_mse))/energies.std(), batch_size=batch_size)
        self.log("val_forces_%rmse", torch.sqrt(torch.clone(forces_val_mse))/forces.std(), batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        feats, properties, systems = batch

        batch_size = len(systems)

        outputs = self(feats, systems)

        outputs = self.energy_transformer.inverse_transform(systems, outputs)
        energy_test_mse, forces_test_mse = self.loss_fn.report(outputs, properties)

        loss = energy_test_mse + forces_test_mse 
        self.log('test_loss', torch.clone(loss))

        self.log("test_energy_mse", torch.clone(energy_test_mse), batch_size = batch_size)
        self.log("test_forces_mse", torch.clone(forces_test_mse), batch_size = batch_size)

        return loss

    def backward(self, loss, *args, **kwargs):
        loss.backward(retain_graph=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1., line_search_fn="strong_wolfe")
        return optimizer