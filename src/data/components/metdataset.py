from typing import Tuple, Callable

import os
import csv
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

class MetDataset(Dataset):
    def __init__(self, data_dir: str, transform: Callable):

        self.data_dir = data_dir
        self.transform = transform

        self.csv_path = os.path.join(self.data_dir, f"EIB2025/dataset.csv")
        self.texts = []
        self.labels = []
        self.datetime = []
        self.npy_paths = []

        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                self.texts.append(row["description"])
                self.datetime.append(row["datetime"])
                self.labels.append(row["label"])
                self.npy_paths.append(row["path"])

        self.npy_paths = [os.path.join(self.data_dir, x)
                          for x in self.npy_paths]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        met_data = np.load(self.npy_paths[idx]).astype(np.float32)
        if self.transform:
            met_data = self.transform(met_data)

        met_data = torch.tensor(met_data)

        label = self.labels[idx]
        datetime = self.datetime[idx]
        text = self.texts[idx]
        return met_data, text, label, datetime


    def __len__(self):
        return len(self.npy_paths)


# if __name__ == "__main__":
#     '/home/joshua/Documents/phd_university/hackathon/EIHackthon2025/'
#     dataset = MetDataset('/home/joshua/Documents/phd_university/hackathon/EIHackthon2025/', transform=None)
#     print(len(dataset.labels))
#     print(len(dataset.texts))
#     print(len(dataset.npy_paths))
