import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))

from nn.model import BPNNModel
from nn.loss import EnergyForceLoss

class BPNNRascalineModule(pl.LightningModule):
    def __init__(self, example_tensormap, energy_transformer):
        super().__init__()
        self.model = BPNNModel()
        self.model.initialize_weights(example_tensormap)
        self.loss_fn = EnergyForceLoss(w_forces=True, force_weight=0.95)
        self.energy_transformer = energy_transformer

    def forward(self, feats, systems):
        return self.model(feats, systems)
    

    def training_step(self, batch, batch_idx):
        feats, properties, systems = batch

        #print(feats.block(0).values.requires_grad)

        properties = self.energy_transformer.transform(systems, properties)

        outputs = self(feats,systems)
        loss = self.loss_fn(outputs, properties)
        print("computing loss")
        self.log('train_loss', loss, enable_graph=True)
        return loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):



        feats, properties, systems = batch

        outputs = self(feats, systems)
        #outputs = self.energy_transformer.inverse_transform(systems, outputs)

        energy_val_mse, forces_val_mse = self.loss_fn.report(outputs, properties)

        loss = energy_val_mse + forces_val_mse

        #self.log('val_loss', loss.item())

        #self.log("val_energy_mse", torch.clone(energy_val_mse))
        #self.log("val_forces_mse", torch.clone(forces_val_mse))

        return loss

    def test_step(self, batch, batch_idx):
        feats, properties, systems = batch
        outputs = self(feats, systems)

        #outputs = self.energy_transformer.inverse_transform(systems, outputs)
        energy_test_mse, forces_test_mse = self.loss_fn.report(outputs, properties)

        loss = energy_test_mse + forces_test_mse 
        #self.log('test_loss', torch.clone(loss))

        #self.log("test_energy_mse", torch.clone(energy_test_mse))
        #self.log("test_forces_mse", torch.clone(forces_test_mse))

        return loss

    def backward(self, loss, *args, **kwargs):
        loss.backward(retain_graph=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer