import torch
import torch.nn as nn
import numpy as np 
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn.functional as F

class TF_Runner(LightningModule):
    def __init__(self, model, lr, weight_decay, T_0, vocab_size):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_0 = T_0
        self.vocab_size = vocab_size
        self.pad = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)

    def forward(self, source, target):
        return self.model(source, target)

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0= self.T_0,
            eta_min = 1e-6
        )
        # scheduler = CosineAnnealingLR(
        #     optimizer=opt, 
        #     T_max= 1000
        # )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss' # Metric to monitor
        }
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    
    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return {"val_loss": loss}

    def validation_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {"val_loss": val_loss,},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss,}

    def test_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return {"val_loss": loss}

    def test_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "test_loss": val_loss,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        result = {"test_loss": float(val_loss.detach().cpu())}
        self.test_results = result
        return result
        

class Runner(LightningModule):
    def __init__(self, model, lr, weight_decay, T_0, vocab_size):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_0 = T_0
        self.vocab_size = vocab_size
        self.pad = 0

    def forward(self, source, target):
        return self.model(source, target)

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0= self.T_0,
            eta_min = 1e-6
        )
        # scheduler = CosineAnnealingLR(
        #     optimizer=opt, 
        #     T_max= 1000
        # )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss' # Metric to monitor
        }
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        source, target = batch
        source = source.transpose(1,0)
        target = target.transpose(1,0)
        output = self.forward(source, target)
        loss = F.nll_loss(output[1:].view(-1, self.vocab_size),
                            target[1:].contiguous().view(-1),
                            ignore_index = self.pad)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    
    def validation_step(self, batch, batch_idx):
        source, target = batch
        source = source.transpose(1,0)
        target = target.transpose(1,0)
        output = self.forward(source, target)
        loss = F.nll_loss(output[1:].view(-1, self.vocab_size),
                            target[1:].contiguous().view(-1),
                            ignore_index= self.pad)
        return {"val_loss": loss}

    def validation_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {"val_loss": val_loss,},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss,}

    def test_step(self, batch, batch_idx):
        source, target = batch
        source = source.transpose(1,0)
        target = target.transpose(1,0)
        output = self.forward(source, target)
        loss = F.nll_loss(output[1:].view(-1, self.vocab_size),
                            target[1:].contiguous().view(-1),
                            ignore_index= self.pad)
        return {"val_loss": loss}

    def test_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
                {
                    "test_loss": val_loss,
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        result = {"test_loss": float(val_loss.detach().cpu())}
        self.test_results = result
        return result
        