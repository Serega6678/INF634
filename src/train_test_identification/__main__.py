import argparse
import os.path
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


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")


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


def transform_path(data: tp.Tuple[str, bool]) -> tp.Tuple[str, str]:
    return transform_path_to_normal_lfw(data), transform_path_to_masked_lfw(data)


def transform_path_to_masked_lfw(data: tp.Tuple[str, bool]) -> str:
    path, face_is_open = data
    path = path.split("/")
    name, _ = path[-1].split(".")
    name += "_fake"
    filename = ".".join([name, "png"])
    if face_is_open:
        path = "/".join([*path[:-5], "data_transformed", "no_mask_to_mask", "images", filename])
        return path
    else:
        path = "/".join([*path[:-4], "data_transformed", "no_mask_to_mask", "images", filename])
        return path


def transform_path_to_normal_lfw(data: tp.Tuple[str, bool]) -> str:
    path, face_is_open = data
    path = path.split("/")
    name, _ = path[-1].split(".")
    name += "_fake"
    filename = ".".join([name, "png"])
    if face_is_open:
        path = "/".join([*path[:-5], "data_transformed", "mask_to_no_mask", "images", filename])
        return path
    else:
        path = "/".join([*path[:-4], "data_transformed", "mask_to_no_mask", "images", filename])
        return path


def get_paths_exist_mask(paths: tp.List[str]) -> np.ndarray:
    exists = [os.path.exists(path) for path in tqdm(paths)]
    return np.array(exists, dtype=bool)


def calculate_default_accuracy():
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


def calculate_print_normal_lfw_accuracy() -> None:
    test_dataset = TestDataset(normal_paths, transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)
    preds = trainer.predict(model, test_dataloader, return_predictions=True)
    preds = torch.concat(preds)
    print(f"Accuracy for normal-LFW full: {accuracy_score(targets[normal_paths_exist_mask], preds.argmax(-1)) * 100}%")
    preds_normal_lfw = preds[~face_is_open[normal_paths_exist_mask]]
    targets_normal_lfw = targets[normal_paths_exist_mask][~face_is_open[normal_paths_exist_mask]]
    preds_normal_lfw = preds_normal_lfw.argmax(-1)
    print(f"Accuracy for normal-LFW: {accuracy_score(targets_normal_lfw, preds_normal_lfw) * 100}%")


def calculate_print_masked_lfw_accuracy() -> None:
    test_dataset = TestDataset(masked_paths, transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)
    preds = trainer.predict(model, test_dataloader, return_predictions=True)
    preds = torch.concat(preds)
    print(f"Accuracy for masked-LFW full: {accuracy_score(targets[masked_paths_exist_mask], preds.argmax(-1)) * 100}%")
    preds_masked_lfw = preds[~face_is_open[masked_paths_exist_mask]]
    targets_masked_lfw = targets[masked_paths_exist_mask][~face_is_open[masked_paths_exist_mask]]
    preds_masked_lfw = preds_masked_lfw.argmax(-1)
    print(f"Accuracy for masked-LFW: {accuracy_score(targets_masked_lfw, preds_masked_lfw) * 100}%")


def calculate_print_cycle_lfw_accuracy() -> None:
    test_dataset = TestDataset(cycle_paths, transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)
    preds = trainer.predict(model, test_dataloader, return_predictions=True)
    preds = torch.concat(preds)
    print(f"Accuracy for cycle-LFW full: {accuracy_score(targets[masked_paths_exist_mask], preds.argmax(-1)) * 100}%")
    preds_masked_lfw = preds[~face_is_open[masked_paths_exist_mask]]
    targets_masked_lfw = targets[masked_paths_exist_mask][~face_is_open[masked_paths_exist_mask]]
    preds_masked_lfw = preds_masked_lfw.argmax(-1)
    print(f"Accuracy for cycle-LFW: {accuracy_score(targets_masked_lfw, preds_masked_lfw) * 100}%")


if __name__ == "__main__":
    seed_everything(42)
    args = parser.parse_args()
    datamodule = DataModulePL("./data")
    print("Number of classes:", datamodule.base_dataset.num_classes)
    if args.train:
        model = ModulePL(datamodule.base_dataset.num_classes)
    else:
        model = ModulePL.load_from_checkpoint("./checkpoints/last.ckpt", out_dim=datamodule.base_dataset.num_classes)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath="./checkpoints",
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
        max_epochs=20 if args.train else 0,
    )
    trainer.fit(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    paths, targets, face_is_open = get_dataset_data(test_dataloader.dataset)

    calculate_default_accuracy()

    transformed_paths = list(map(transform_path, zip(paths, face_is_open)))
    normal_paths, masked_paths = list(zip(*transformed_paths))

    normal_paths_exist_mask = get_paths_exist_mask(normal_paths)
    masked_paths_exist_mask = get_paths_exist_mask(masked_paths)

    normal_paths = list(map(lambda x: x[0], filter(lambda x: x[1], zip(normal_paths, normal_paths_exist_mask))))
    masked_paths = list(map(lambda x: x[0], filter(lambda x: x[0], zip(masked_paths, masked_paths_exist_mask))))
    cycle_paths = list(map(lambda x: x.replace("no_mask_to_mask", "cycle_transform").replace("fake", "fake_fake"), masked_paths))

    calculate_print_normal_lfw_accuracy()
    calculate_print_masked_lfw_accuracy()
    calculate_print_cycle_lfw_accuracy()
