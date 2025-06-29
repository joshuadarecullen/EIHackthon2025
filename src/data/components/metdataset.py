from typing import Tuple, Callable

import os
import csv
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

"""
A dataset class that loads the climate data npy files, with its correspinding
label, text description, and date-time it was captured. With the last three
variables being loaded in from the dataset.csv file.

Created by: Joshua Dare-Cullen

"""
class MetDataset(Dataset):
    def __init__(self, data_dir: str, transform: Callable):

        self.data_dir = data_dir
        self.transform = transform

        # either make sure dataset csv is in the data/EIB2025, or set path
        try:
            self.csv_path = os.path.join(self.data_dir, f"EIB2025/dataset.csv")
        except Exception as e:
            print(f'File does not exist')

        self.texts = []
        self.labels = []
        self.datetime = []
        self.npy_paths = []
        self.label_idx: dict = None

        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                self.texts.append(row["description"])
                self.datetime.append(row["datetime"])
                self.labels.append(row["label"])
                self.npy_paths.append(row["path"])

        self.npy_paths = [os.path.join(self.data_dir, x)
                          for x in self.npy_paths]

        self.labels = self._onehot(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        met_data = np.load(self.npy_paths[idx]).astype(np.float32)
        if self.transform:
            met_data = self.transform(met_data)
        met_data = torch.tensor(met_data)

        label = self.labels[idx]
        datetime = self.datetime[idx]
        text = self.texts[idx]
        return {
                "climate_data": met_data,
                "label": label,
                "text": self.texts[idx],       # optional string
                "datetime": self.datetime[idx] # optional string
                }

    def __len__(self) -> int:
        return len(self.npy_paths)

    def _onehot(self, label_list) -> torch.Tensor:

        # Get unique classes and build label-to-index map
        unique_classes = sorted(set(label_list))  # ensure deterministic order
        class_to_idx = {label: idx for idx, label in enumerate(unique_classes)}

        # we can use this dictionrary to map back to the text label from label indices
        self.id_to_text_label = {idx: label for idx, label in enumerate(unique_classes)}

        # Map labels to indices
        label_indices = torch.tensor([class_to_idx[label] for label in label_list])

        # Step 4: One-hot encode
        num_classes = len(unique_classes)
        one_hot_labels = F.one_hot(label_indices, num_classes=num_classes).float()
        labels_indices = torch.argmax(one_hot_labels, dim=1)  # shape: [428]
        return label_indices
