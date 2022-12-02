import typing as tp
from tqdm import tqdm
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader

from src.pl import ModulePL, DataModulePL
from src.data import TestDataset, test_transforms


def get_dataset_data(dataset: Dataset) -> tp.Tuple[tp.List[str], np.ndarray, np.ndarray]:
    paths = []
    targets = []
    face_is_open = []
    for item in tqdm(dataset, desc="Obtaining dataset data"):
        paths.append(item["path"])
        targets.append(item["face_idx"])
        face_is_open.append(item["face_type"])
    targets = np.array(targets)
    face_is_open = np.array(face_is_open, dtype=bool)
    return paths, targets, face_is_open


def transform_path(data: tp.Tuple[str, bool]) -> str:
    path, face_is_open = data
    path = path.split("/")
    name, _ = path[-1].split(".")
    name += "_fake"
    filename = ".".join([name, "png"])
    if face_is_open:
        path = "/".join([*path[:-5], "data_transformed", "no_mask_to_mask", "images", filename])
        return path
    else:
        path = "/".join([*path[:-4], "data_transformed", "mask_to_no_mask", "images", filename])
        return path


if __name__ == "__main__":
    seed_everything(42)
    datamodule = DataModulePL("./data")
    print("Number of classes:", datamodule.base_dataset.num_classes)
    model = ModulePL(datamodule.base_dataset.num_classes)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath="./checkpoints/FaceID",
                monitor="val_accuracy_epoch",
                filename="{epoch}-{val_accuracy_epoch:.4f}",
                mode="max",
                save_last=True
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=WandbLogger(project="INF634_FaceId"),
        num_sanity_val_steps=2,
        log_every_n_steps=10,
        max_epochs=20,
    )
    trainer.fit(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    paths, targets, face_is_open = get_dataset_data(test_dataloader.dataset)

    test_dataset = TestDataset(paths, transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)

    preds = trainer.predict(model, test_dataloader, return_predictions=True)
    preds = torch.concat(preds)

    preds_open_face, targets_open_face = preds[face_is_open], targets[face_is_open]

    preds_open_face = preds_open_face.argmax(-1)

    print(f"Accuracy for open faces: {accuracy_score(targets_open_face, preds_open_face) * 100}%")

    preds_close_face, targets_close_face = preds[~face_is_open], targets[~face_is_open]
    preds_close_face = preds_close_face.argmax(-1)
    print(f"Accuracy for closed faces: {accuracy_score(targets_close_face, preds_close_face) * 100}%")

    transformed_paths = list(map(transform_path, zip(paths, face_is_open)))
    test_dataset = TestDataset(transformed_paths, transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)

    preds = trainer.predict(model, test_dataloader, return_predictions=True)
    preds = torch.concat(preds)

    preds_masked_lfw, targets_masked_lfw = preds[face_is_open], targets[face_is_open]
    preds_masked_lfw = preds_masked_lfw.argmax(-1)
    print(f"Accuracy for masked-LFW: {accuracy_score(targets_masked_lfw, preds_masked_lfw) * 100}%")

    preds_normal_lfw, targets_normal_lfw = preds[~face_is_open], targets[~face_is_open]
    preds_normal_lfw = preds_normal_lfw.argmax(-1)
    print(f"Accuracy for normal-LFW: {accuracy_score(targets_normal_lfw, preds_normal_lfw) * 100}%")
