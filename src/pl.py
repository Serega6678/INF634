import typing as tp

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torchmetrics import Accuracy, AUROC
import pytorch_lightning as pl

from src.data import LFWDataset, transforms


class ModulePL(pl.LightningModule):
    def __init__(self, out_dim: int) -> None:
        super(ModulePL, self).__init__()
        self.model = nn.Sequential(
            torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True),
            nn.Linear(1000, out_dim)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_roc_auc = AUROC(num_classes=out_dim)
        self.val_roc_auc = AUROC(num_classes=out_dim)

    def forward(self, imgs: torch.Tensor) -> tp.Any:
        return self.model(imgs)

    def training_step(self, train_batch: tp.Dict[str, tp.Any], batch_idx: tp.Any) -> torch.Tensor:
        imgs = train_batch["img"]
        target = train_batch["face_idx"]
        output = self.forward(imgs)
        loss = self.criterion(output, target)
        preds = output.argmax(-1)
        self.train_acc(preds, target)
        self.train_roc_auc(output, target)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_acc)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_accuracy_epoch", self.train_acc)
        self.log("train_rocauc_epoch", self.train_roc_auc)

    def validation_step(self, val_batch: tp.Dict[str, tp.Any], batch_idx: tp.Any) -> None:
        imgs = val_batch["img"]
        target = val_batch["face_idx"]
        output = self.forward(imgs)
        loss = self.criterion(output, target)
        preds = output.argmax(-1)
        self.val_acc(preds, target)
        self.val_roc_auc(output, target)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_acc)

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_accuracy_epoch", self.val_acc)
        self.log("val_rocauc_epoch", self.val_roc_auc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DataModulePL(pl.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 32) -> None:
        super(DataModulePL, self).__init__()
        self.path = path
        self.base_dataset = LFWDataset(path=self.path, transform=transforms())

        self.train_dataset = None
        self.val_dataset = None

        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        train_size = len(self.base_dataset) // 4 * 3
        val_size = len(self.base_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.base_dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
