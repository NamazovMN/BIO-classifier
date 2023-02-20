import torch
from torch.utils.data import Dataset
from process_data import ProcessData

from tqdm import tqdm


class ArgumentDataset(Dataset):
    """
    Dataset object which collects and defines required dataset
    """

    def __init__(self, config_parameters: dict, task_name: str, ds_type: str):
        """
        Method performs as an initializer of the class
        :param config_parameters: configuration parameters of the project
        :param task_name: name of the sub-task, which can be either BIO or POS
        :param ds_type: dataset type which can be either of train, dev, test
        """
        self.data, self.labels = self.collect_dataset(config_parameters, task_name, ds_type)

    @staticmethod
    def collect_dataset(parameters: dict, task_name: str, ds_type: str):
        """
        Method is utilized to collect required data for the dataset
        :param parameters: configuration parameters of the project
        :param task_name: name of the sub-task, which can be either BIO or POS
        :param ds_type: dataset type which can be either of train, dev, test
        :return: tuple of data tensor and label tensor
        """
        process = ProcessData(parameters, task_name, ds_type)
        data = list()
        labels = list()
        ti = tqdm(process, total=process.__len__(), desc=f'Dataset is collected for {task_name} {ds_type}', leave=True)
        for sentences, targets in ti:
            data.extend(sentences)
            labels.extend(targets)

        return torch.LongTensor(data), torch.LongTensor(labels)

    def __getitem__(self, item: int) -> dict:
        """
        Method performs as a getter of the object
        :param item: idx of the data is requested
        :return: dictionary data which contains data and label for corresponding data
        """
        return {
            'data': self.data[item],
            'label': self.labels[item]
        }

    def __len__(self) -> int:
        """
        Method is utilized to get the length of the dataset
        :return: number of data in the dataset
        """
        return len(self.data)
