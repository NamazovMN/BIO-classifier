import torch
import numpy as np
from torch.utils.data import Dataset

class TokensDataset(Dataset):
    def __init__(self, data, labels):
        n_data, n_labels = self.generate_dataset(data, labels)
        self.data = torch.LongTensor(np.array(n_data))
        self.labels = torch.LongTensor(np.array(n_labels))

    def generate_dataset(self, data, labels):
        new_data = list()
        new_labels = list()
        for each_data, each_label in zip(data, labels):
            new_data.extend(each_data)
            new_labels.extend(each_label)
        return new_data, new_labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'labels': self.labels[idx]
        }