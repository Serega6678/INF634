import os
import shutil
import sys

from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from data import LFWDataset


if __name__ == "__main__":
    dataset_path, output_path = sys.argv[1:]
    dataset = LFWDataset(dataset_path)
    gss = GroupShuffleSplit(n_splits=1, random_state=42)
    for train_idx, test_idx in gss.split(dataset.face_paths, dataset.face_paths, groups=dataset.face_ids):
        train_paths = [dataset.face_paths[i] for i in train_idx]
        train_masks = [dataset.face_is_open[i] for i in train_idx]

        test_paths = [dataset.face_paths[i] for i in test_idx]
        test_masks = [dataset.face_is_open[i] for i in test_idx]

        for path, face_type in tqdm(zip(train_paths, train_masks), total=len(train_paths)):
            filename = path.split("/")[-1]

            dst = output_path + "/" + ("trainA" if face_type else "trainB")

            if not os.path.exists(dst):
                os.makedirs(dst)

            dst += "/" + filename

            shutil.copyfile(path, dst)

        for path, face_type in tqdm(zip(test_paths, test_masks), total=len(test_paths)):
            filename = path.split("/")[-1]

            dst = output_path + "/" + ("testA" if face_type else "testB")

            if not os.path.exists(dst):
                os.makedirs(dst)

            dst += "/" + filename

            shutil.copyfile(path, dst)
        break
