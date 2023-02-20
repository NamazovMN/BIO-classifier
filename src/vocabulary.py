import os
import pickle
from src.read_data import ReadData

class Vocabulary:
    def __init__(self, parameters, task_name, ds_type):
        self.configuration = self.set_configuration(parameters, task_name, ds_type)
        self.vocabulary, self.lab2id = self.get_vocabulary() if task_name == 'BIO' else self.get_pos_labels()

    def set_configuration(self, parameters, task_name, ds_type):
        result_parameters = parameters
        result_parameters['processed_dir'] = os.path.join(parameters['input_dir'], 'processed_data')
        result_parameters['task_name'] = task_name
        result_parameters['ds_type'] = ds_type
        return result_parameters

    def get_vocabulary(self):
        vocab_path = os.path.join(self.configuration['processed_dir'], 'vocabulary.pickle')
        lab2id_bio_path = os.path.join(self.configuration['processed_dir'], 'lab2id_bio.pickle')

        if not os.path.exists(vocab_path):
            if self.configuration['ds_type'] != 'train':
                raise FileNotFoundError('Vocabulary was not found! Reason: You are trying to build vocabulary based on'
                                        ' other datasets but train. Please check your order!')
            reader = ReadData(self.configuration, 'train', 'BIO')
            sentences = list()
            labels = list()
            for sentence, targets in reader:
                sentences.extend(sentence)
                labels.extend(targets)
            vocabulary = {token: idx for idx, token in enumerate(set(sentences))}
            vocabulary['<UNK>'] = len(vocabulary)
            vocabulary['<PAD>'] = len(vocabulary)
            lab2id = {label: idx for idx, label in enumerate(set(labels))}
            lab2id['<PAD>'] = len(lab2id)
            with open(vocab_path, 'wb') as vocab_data:
                pickle.dump(vocabulary, vocab_data)
            with open(lab2id_bio_path, 'wb') as lab2id_bio:
                pickle.dump(lab2id, lab2id_bio)
        with open(vocab_path, 'rb') as vocab_data:
            vocabulary = pickle.load(vocab_data)
        with open(lab2id_bio_path, 'rb') as lab2id_bio:
            lab2id = pickle.load(lab2id_bio)
        return vocabulary, lab2id

    def get_pos_labels(self):
        lab2id_pos_path = os.path.join(self.configuration['processed_dir'], 'lab2id_pos.pickle')
        vocab_path = os.path.join(self.configuration['processed_dir'], 'vocabulary.pickle')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError('Vocabulary does not exist. You get this error, because you try to build vocabulary'
                                    ' according to the secondary task: POS. Change the order!')
        if not os.path.exists(lab2id_pos_path):
            types = ['train', 'test']
            labels = list()
            for each_type in types:
                reader = ReadData(self.configuration, each_type, 'POS')
                for _, targets in reader:
                    labels.extend(targets)
            lab2id = {label: idx for idx, label in enumerate(set(labels))}
            lab2id['<PAD>'] = len(lab2id)
            with open(lab2id_pos_path, 'wb') as lab2id_pos_data:
                pickle.dump(lab2id, lab2id_pos_data)
        with open(lab2id_pos_path, 'rb') as lab2id_pos_data:
            lab2id = pickle.load(lab2id_pos_data)
        with open(vocab_path, 'rb') as vocab_data:
            vocabulary = pickle.load(vocab_data)
        return vocabulary, lab2id

    def __getitem__(self, item):
        return self.vocabulary[item] if item in self.vocabulary.keys() else self.vocabulary['<UNK>']

    def __len__(self):
        return len(self.vocabulary)