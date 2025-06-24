from typing import Tuple, Path, Callable

import csv
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from PIL import Image

import clip
from transformers import CLIPProcessor


class MetDataset(Dataset):
    def __init__(self, data_dir: Path, transform: Callable):

        self.data_dir = data_dir
        self.transform = Callable[Tensor, Tensor]]

        self.csv_path = os.path.join(self.data_dir, f"{metdata}.csv")
        texts = []
        met_data_paths = []

        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                texts.append(row["text"])
                met_data_paths.append(row["met_data_path"])

        self.met_data_paths = [os.path.join(self.data_dir, x)
                          for x in met_data_paths]
        self.texts = texts

        # self.tokenized_texts = clip.tokenize(texts)

        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preprocess met data using CLIP's preprocessing function
        # met_data_sample = preprocess(np.load(self.met_data_paths[idx]))
        # met_data_sample = np.load(self.met_data_paths[idx])
        text = self.texts[idx]
        return met_data_sample, text


    def __len__(self):
        return len(self.met_data_paths)
