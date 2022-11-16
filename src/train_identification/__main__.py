from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.pl import ModulePL, DataModulePL

if __name__ == "__main__":
    datamodule = DataModulePL("./data")
    model = ModulePL(datamodule.base_dataset.num_classes)
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath="./checkpoints/FaceID", monitor="val_loss")],
        logger=WandbLogger(project="INF634_FaceId"),
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule)
