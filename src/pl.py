import typing as tp

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy, AUROC
import pytorch_lightning as pl

from src.data import LFWDataset, transforms


class ModulePL(pl.LightningModule):
    def __init__(self, out_dim: int, backbone: tp.Optional[str] = None, hub: tp.Optional[str] = None) -> None:
        super(ModulePL, self).__init__()
        if backbone is None:
            backbone = "resnet18"
        if hub is None:
            hub = "pytorch/vision:v0.10.0"
        self.model = nn.Sequential(
            torch.hub.load(hub, backbone, pretrained=True),
            nn.Linear(1000, out_dim)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_roc_auc = AUROC(num_classes=out_dim)
        self.val_roc_auc = AUROC(num_classes=out_dim)
        self.test_roc_auc = AUROC(num_classes=out_dim)

    def forward(self, batch: tp.Dict[str, tp.Any]) -> tp.Any:
        return self.model(batch["img"])

    def training_step(self, train_batch: tp.Dict[str, tp.Any], batch_idx: tp.Any) -> torch.Tensor:
        target = train_batch["face_idx"]
        output = self.forward(train_batch)
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
        target = val_batch["face_idx"]
        output = self.forward(val_batch)
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
        scheduler = MultiStepLR(optimizer, [8, 14], 0.1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class DataModulePL(pl.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 32) -> None:
        super(DataModulePL, self).__init__()
        self.path = path
        self.base_dataset = LFWDataset(path=self.path, transform=transforms())

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        train_size = int(len(self.base_dataset) * 0.6)
        val_size = (len(self.base_dataset) - train_size) // 2
        test_size = len(self.base_dataset) - train_size - val_size
        assert train_size + val_size + test_size == len(self.base_dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.base_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)
