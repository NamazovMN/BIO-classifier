import pandas as pd
from utilities import *
import pickle
from tqdm import tqdm
from nltk import pos_tag
class ReadData:
    def __init__(self, config_parameters, ds_type, task_name):
        self.configuration = self.set_configuration(config_parameters, ds_type, task_name)
        self.tokens = self.read_lines()
        self.dataset = self.collect_sentences()

    def set_configuration(self, parameters, ds_type, task_name):
        input_dir = parameters['input_dir']
        processed_dir = os.path.join(input_dir, 'processed_data')
        check_dir(processed_dir)
        process_type = parameters['process_type']
        return {
            'input_dir': input_dir,
            'processed_dir': processed_dir,
            'task_name': task_name,
            'data_sep': '__END_PARAGRAPH__' if process_type == 'paragraph' else '__END_ESSAY__',
            'process_type': process_type,
            'ds_type': ds_type,
            'delimiters': ['__END_PARAGRAPH__', '__END_ESSAY__', '.', '?', '!']
        }

    def read_lines(self):
        input_data = os.path.join(self.configuration['input_dir'], f'{self.configuration["ds_type"]}-bio.csv')
        dataset = {'token': list(), 'label': list()}
        for each_line in open(input_data, 'r'):
            current_line = each_line.split('\t')
            dataset['token'].append(current_line[0])
            dataset['label'].append(current_line[1].replace('\n', ''))
        return dataset

    def tag_sentence(self, current_sentence):
        sentence = [token['token'] for token in current_sentence]
        if len(sentence) == 1:
            result_sentence = [{'token': sentence[0], 'label': '<EOP>' if sentence[0] == '__END_OF_PARAGRAPH__' else '<EOE>'}]
        else:
            tagged = pos_tag(sentence)
            result_sentence = [{'token': token[0], 'label': token[1]} for token in tagged]
        return result_sentence

    def collect_sentences(self):
        tokens = self.read_lines()

        data = {
            'sentences': list(),
            'labels': list()
        }
        current_sentence = list()
        task_info = '_POS' if self.configuration['task_name'] == 'POS' else ''
        file_path = os.path.join(self.configuration['processed_dir'], f'sentences_{self.configuration["ds_type"]}{task_info}.pickle')
        if not os.path.exists(file_path):
            ti = tqdm(enumerate(zip(tokens['token'], tokens['label'])), total=len(tokens['token']),
                      desc='Tokens are analyzed for sentence', leave=True)
            for idx, (token, label) in ti:
                current_sentence.append({'token': token, 'label': label})
                if token in self.configuration['delimiters']:
                    if self.configuration['task_name'] == 'POS':
                        current_sentence = self.tag_sentence(current_sentence)
                    data['sentences'].append([token['token'] for token in current_sentence])
                    data['labels'].append([token['label'] for token in current_sentence])
                    current_sentence = list()
            with open(file_path, 'wb') as sentence_data:
                pickle.dump(data, sentence_data)
        with open(file_path, 'rb') as sentence_data:
            data = pickle.load(sentence_data)
        return data



    def __iter__(self):
        for data, labels in zip(self.dataset['sentences'], self.dataset['labels']):
            yield data, labels

    def __len__(self):
        return len(self.dataset['sentences'])
