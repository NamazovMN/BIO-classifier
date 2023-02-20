import torch
from torch.utils.data import Dataset
from src.process_data import ProcessData
from tqdm import tqdm
class BIO_Dataset(Dataset):
    def __init__(self, parameters, ds_type, task_name):
        self.configuration = self.set_configuration(parameters, ds_type, task_name)
        self.dataset, self.targets = self.collect_dataset()

    def set_configuration(self, parameters, ds_type, task_name):
        return {
            'ds_type': ds_type,
            'task_name': task_name,
            'process_obj': ProcessData(parameters, ds_type, task_name)
        }

    def collect_dataset(self):
        dataset = list()
        targets = list()
        ti = tqdm(self.configuration['process_obj'], total=self.configuration['process_obj'].__len__(),
                  desc=f'Dataset is collected for {self.configuration["ds_type"]} data of '
                       f'{self.configuration["task_name"]}')
        for data, labels in ti:

            dataset.extend(data)
            targets.extend(labels)
        return torch.LongTensor(dataset), torch.LongTensor(targets)

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'label': self.targets[item]
        }

    def __len__(self):
        return len(self.dataset)
