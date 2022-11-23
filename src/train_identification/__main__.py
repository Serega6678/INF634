from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.pl import ModulePL, DataModulePL

if __name__ == "__main__":
    seed_everything(42)
    datamodule = DataModulePL("./data")
    print("Number of classes:", datamodule.base_dataset.num_classes)
    model = ModulePL(datamodule.base_dataset.num_classes)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath="./checkpoints/FaceID",
                monitor="val_accuracy",
                filename="{epoch}-{val_accuracy:.4f}-{val_loss:.4f}",
                mode="max",
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=WandbLogger(project="INF634_FaceId"),
        num_sanity_val_steps=2,
        log_every_n_steps=10,
        max_epochs=20,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
