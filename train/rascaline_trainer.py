import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "model"))

from nn.model import BPNNModel
from nn.loss import EnergyForceLoss

class BPNNRascalineModule(pl.LightningModule):
    def __init__(self, example_tensormap):
        super().__init__()
        self.model = BPNNModel()
        self.model.initialize_weights(example_tensormap)
        self.loss_fn = EnergyForceLoss(w_forces=True, force_weight=0.95)

    def forward(self, feats, systems):
        return self.model(feats, systems)

    def training_step(self, batch, batch_idx):
        feats, properties, systems = batch

        outputs = self(feats,systems)
        loss = self.loss_fn(outputs, properties)
        print("computing loss")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, properties, systems = batch
        outputs = self(feats, systems)
        loss = self.loss_fn(outputs, properties)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        feats, properties, systems = batch
        outputs = self(feats, systems)
        loss = self.loss_fn(outputs, properties)
        self.log('test_loss', loss)
        return loss

    def backward(self, loss, *args, **kwargs):
        loss.backward(retain_graph=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer