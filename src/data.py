from pathlib import Path
import typing as tp

import cv2
from torchvision.transforms import Compose, Normalize, ColorJitter, ToTensor, ToPILImage, RandomApply, RandomHorizontalFlip
from torch.utils.data import Dataset


def transforms():
    return Compose([
        ToPILImage(),
        RandomHorizontalFlip(),
        RandomApply([ColorJitter(0.05, 0.05, 0.05)], p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class LFWDataset(Dataset):
    OPEN_FACE_DIR_NAME = "LFW_without_Mask/LFW_without_Mask"
    MASKED_FACE_DIR_NAME = "Masked_LFW_Dataset/Masked_LFW_Dataset"

    def __init__(self, path: str, transform: Compose = None) -> None:
        self.root_path = Path(path)
        self.transform = transform

        self.face_paths = []
        self.face_is_open = []
        self.face_ids = []
        self.name_to_idx = {}

        self._read_data()

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        img = cv2.imread(self.face_paths[idx])
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return {
            "img": img,
            "face_type": self.face_is_open[idx],
            "face_idx": self.face_ids[idx],
        }

    def __len__(self) -> int:
        return len(self.face_paths)

    def _read_data(self) -> None:
        open_faces_dir = self.root_path / self.OPEN_FACE_DIR_NAME
        close_face_dir = self.root_path / self.MASKED_FACE_DIR_NAME

        for filepath in open_faces_dir.glob("./*/*"):
            if not filepath.is_file():
                continue

            person_name = filepath.parent.name
            if person_name not in self.name_to_idx:
                self.name_to_idx[person_name] = len(self.name_to_idx)

            self.face_paths.append(str(filepath.absolute()))
            self.face_ids.append(self.name_to_idx[person_name])
            self.face_is_open.append(1)

        for filepath in close_face_dir.iterdir():
            if not filepath.is_file():
                continue

            person_name = "_".join(filepath.name.split("_")[:-1])

            self.face_paths.append(str(filepath.absolute()))
            self.face_ids.append(self.name_to_idx[person_name])
            self.face_is_open.append(0)

    @property
    def num_classes(self) -> int:
        return len(self.name_to_idx)


if __name__ == "__main__":
    dataset = LFWDataset("data")
