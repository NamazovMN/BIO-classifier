import os
from utilities import *
from read_data import ReadData
import pickle
from vocabulary import Vocabulary


class ProcessData:
    """
    Class is utilized to process the data for the classification task
    """
    def __init__(self, config_parameters, task_name, ds_type):
        """
        Method performs as an initializer of the class
        :param config_parameters: configuration parameters of the project
        :param task_name: name of the sub-task, which can be either BIO or POS
        :param ds_type: dataset type which can be either of train, dev, test
        """
        self.configuration = self.set_configuration(config_parameters, task_name)
        self.dataset = self.collect_dataset()[ds_type]
        self.vocabulary, self.lab2id = self.set_vocabulary()

    @staticmethod
    def set_configuration(parameters: dict, task_name: str) -> dict:
        """
        Method is utilized to extract task-specific parameters from the configuration parameters of the project
        :param parameters: configuration parameters of the project
        :param task_name: name of the task for which the dataset will be processed
        :return: dictionary that contains task-specific parameters
        """
        configuration = parameters
        output_env = os.path.join(parameters['input_dir'], 'model_data')
        split_info = '_with_split' if parameters['split'] else ''
        cased_info = '_cased' if parameters['cased'] else ''
        stopwords_info = '_stops' if parameters['clean_stops'] else ''
        output_dir = os.path.join(output_env, f'dataset{split_info}{stopwords_info}{cased_info}')
        check_dir(output_env)
        check_dir(output_dir)
        configuration['output_dir'] = output_dir
        configuration['result_file'] = os.path.join(output_dir, f'{task_name}_dataset_'
                                                                f'{parameters["process_type"]}.pickle')
        configuration['task_name'] = task_name
        return configuration

    def collect_dataset(self) -> dict:
        """
        Method is utilized to collect training, development and test datasets of the given task for further processes
        :return: dictionary contains all types of datasets for the given task
        """
        dataset = {
            'train': dict(), 'test': dict()
        } if not self.configuration['split'] else {
            'train': dict(), 'test': dict(), 'dev': dict()
        }
        if not os.path.exists(self.configuration['result_file']):

            for each_type in ['train', 'test']:
                reader = ReadData(self.configuration, each_type, self.configuration['task_name'])

                if not self.configuration['split']:
                    dataset[each_type] = reader.dataset
                else:
                    if each_type == 'test':
                        dataset[each_type] = reader.dataset
                    else:
                        sentences = list()
                        targets = list()
                        for idx, (data, labels) in enumerate(reader):
                            sentences.append(data)
                            targets.append(labels)
                            if idx == int(reader.__len__() * self.configuration['split']):
                                dataset['train'] = {
                                    'data': sentences, 'labels': targets
                                }
                                sentences = list()
                                targets = list()
                            else:
                                dataset['dev'] = {
                                    'data': sentences, 'labels': targets
                                }
            with open(self.configuration['result_file'], 'wb') as datasets:
                pickle.dump(dataset, datasets)
        with open(self.configuration['result_file'], 'rb') as datasets:
            dataset = pickle.load(datasets)
        return dataset

    def set_vocabulary(self) -> tuple:
        """
        Method is utilized to set the vocabulary
        :return: tuple which contains vocabulary and dictionary in form of {label: idx}
        """
        vocabulary = Vocabulary(self.configuration)
        return vocabulary, vocabulary.lab2id

    def create_windows(self, data: list, is_label: bool = False) -> list:
        """
        Method is utilized to put provided data sequence in form of windows
        :param data: list of tokens that provided sentence contain
        :param is_label: boolean variable specifies whether label sentence is used or tokens
        :return: list of windows which were created based on input data
        """
        window_data = list()
        for idx in range(0, len(data), self.configuration['window_shift']):
            window = data[idx: idx + self.configuration['window_size']]
            difference = self.configuration['window_size'] - len(window)
            if difference:
                window += ['<PAD>'] * difference
            window_data.append(self.index_data(window, is_label=is_label))
        return window_data

    def index_data(self, data: list, is_label: bool = False):
        """
        Method is used to encode provided data in terms of vocabulary/lab2id indexes
        :param data: data is needed to be encoded
        :param is_label: boolean variable specifies whether labels or tokens are encoded
        :return: list of encoded tokens
        """
        return [self.lab2id[each] if is_label else self.vocabulary[each] for each in data]

    def __iter__(self) -> None:
        """
        Method makes the class to perform as a generator
        :return: Method yields tuple of windows of data and corresponding labels
        """
        for data, labels in zip(self.dataset['data'], self.dataset['labels']):
            yield self.create_windows(data), self.create_windows(labels, is_label=True)

    def __len__(self) -> int:
        """
        Method is utilized to return number of data in processed dataset
        :return: length of the processed data
        """
        return len(self.dataset['data'])
