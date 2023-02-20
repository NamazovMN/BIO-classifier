from src.read_data import ReadData
from src.vocabulary import Vocabulary
from tqdm import tqdm
from utilities import *
import pickle
class ProcessData:
    def __init__(self, parameters, ds_type, task_name):
        self.configuration = self.set_configuration(parameters, ds_type, task_name)
        print(self.configuration['vocabulary'].__len__())
        self.dataset = self.collect_req_type()

    def set_configuration(self, parameters, ds_type, task_name):
        reader_obj = ReadData(parameters, ds_type, task_name)
        model_data = os.path.join(parameters['input_dir'], 'model_data')

        check_dir(model_data)
        vocabulary = Vocabulary(parameters, task_name, ds_type)
        cased_info = '_cased' if parameters['cased'] else ''
        stop_info = '_wostop' if parameters['clean_stops'] else ''
        output_folder = os.path.join(model_data, f"data{cased_info}{stop_info}")
        check_dir(output_folder)
        return {
            'ds_type': ds_type,
            'cased_info': cased_info,
            'stop_info': stop_info,
            'output_dir': output_folder,
            'reader': reader_obj,
            'model_data': model_data,
            'task_name': task_name,
            'vocabulary': vocabulary,
            'lab2id': vocabulary.lab2id,
            'delimiter': '__END_PARAGRAPH__' if parameters['process_type'] else '__END_ESSAY__',
            'window_size': parameters['window_size'],
            'window_shift': parameters['window_shift']
        }

    def clean_index(self, sentence):
        tokens = list()
        labels = list()
        sentence_new = [token['token'] for token in sentence]
        for token in sentence:
            tokens.append(self.configuration['vocabulary'][token['token']])
            labels.append(self.configuration['lab2id'][token['label']])
        print(sentence_new)
        return tokens, labels


    def get_clean_data(self, sentence):
        tokens = list()
        labels = list()
        for token in sentence:
            tokens.append(token['token'])
            labels.append(token['label'])
        return tokens, labels

    def collect_req_type(self):
        raw_data = list()
        raw_labels = list()
        dataset = {
            'data': list(),
            'labels': list()
        }
        file_path = os.path.join(self.configuration['output_dir'], f"{self.configuration['task_name']}_{self.configuration['ds_type']}_{se,f}.pickle")
        if not os.path.exists(file_path):
            ti = tqdm(self.configuration['reader'], total=self.configuration['reader'].__len__(), desc=f"Data is collected for {self.configuration['ds_type']} data of {self.configuration['task_name']}")
            for sentence_data, targets_data in ti:
                # sentence, labels = self.clean_index(each_data)
                if sentence_data[0] != self.configuration['delimiter'] and len(sentence_data) == 1:
                    pass
                else:

                    raw_data.extend(sentence_data)
                    raw_labels.extend(targets_data)
                if sentence_data[0] == self.configuration['delimiter']:
                    dataset['data'].append(self.create_windows(raw_data, is_label=False))
                    dataset['labels'].append(self.create_windows(raw_labels))
            with open(file_path, 'wb') as ds_dir:
                pickle.dump(dataset, ds_dir)
        with open(file_path, 'rb') as ds_dir:
            dataset = pickle.load(ds_dir)
        return dataset

    def index_window(self, data, is_label):
        vocab = self.configuration['lab2id'] if is_label else self.configuration['vocabulary']
        return [vocab[token] for token in data]

    def create_windows(self, input_data, is_label=True):
        result_data = list()
        for idx in range(0, len(input_data), self.configuration['window_shift']):
            window = input_data[idx: idx + self.configuration['window_size']]
            difference = self.configuration['window_size'] - len(window)
            if difference:
                window += ['<PAD>'] * difference
            result_data.append(self.index_window(window, is_label))
        return result_data

    def __iter__(self):
        for sentence, labels in zip(self.dataset['data'], self.dataset['labels']):
            yield sentence, labels

    def __len__(self):
        return len(self.dataset['data'])
