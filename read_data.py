import collections
import os
from utilities import *
from nltk import pos_tag
import pickle
from tqdm import tqdm


class ReadData:
    """
    Class is utilized to read and create specific dataset according to the choice of user which can be:
    sentence, paragraph or essay
    """

    def __init__(self, config_parameters: dict, ds_type: str, task_name: str):
        """
        Method performs as an initializer of the class
        :param config_parameters: configuration parameters of the project
        :param task_name: name of the sub-task, which can be either BIO or POS
        :param ds_type: dataset type which can be either of train, dev, test
        """
        self.configuration = self.set_configuration(config_parameters, ds_type, task_name)
        self.dataset = self.process_data()

    def set_configuration(self, parameters: dict, ds_type: str, task_name: str) -> dict:
        """
        Method is utilized to extract task-specific parameters from the configuration parameters of the project
        :param parameters: configuration parameters of the project
        :param ds_type: type of dataset will be read and collected
        :param task_name: name of the task for which the dataset will be processed
        :return: dictionary that contains task-specific parameters
        """
        processed_dir = os.path.join(parameters['input_dir'], 'processed_data')
        stopwords_info = '_stops' if parameters['clean_stops'] else ''
        case_info = '_cased' if parameters['cased'] else ''
        check_dir(processed_dir)
        result_file = os.path.join(processed_dir, f'{task_name}_{ds_type}_{parameters["process_type"]}'
                                                  f'{stopwords_info}{case_info}.pickle')
        return {
            'source_dir': os.path.join(parameters['input_dir'], f'{ds_type}-bio.csv'),
            'task_name': task_name,
            'ds_type': ds_type,
            'cased': parameters['cased'],
            'stops': parameters['clean_stops'],
            'result_file': result_file,
            'processed_dir': processed_dir,
            'process_type': parameters['process_type'],
            'sentence_collect': self.collect_sentences(),
            'paragraph_collect': self.collect_paragraphs(),
            'essay_collect': self.collect_essays()

        }

    def collect_tokens(self) -> collections.Iterable:
        """
        Method is utilized to retrieve tokens from the given dataset
        :return: Method yields dictionary element which contains specific token and its label
        """
        for each_line in open(self.configuration['source_dir'], 'r'):
            token, label = each_line.split(sep='\t')
            yield {'token': token, 'label': label.replace('\n', '')}

    @staticmethod
    def get_pos_tags(data: list) -> list:
        """
        Method is utilized to extract POS tag features for the provided sequence
        :param data: input data which is the list of token dictionaries
        :return: POS features of the provided sequence
        """
        sentence = [token['token'] for token in data]
        result_sentence = pos_tag(sentence)
        labels = [token['label'] for token in data]
        assert (len(sentence) == len(result_sentence))
        pos_tagged = [{'token': tagged[1], 'label': label} for tagged, label in zip(result_sentence, labels)]
        return pos_tagged

    def clean_sentence(self, source_sentence: list, result_sentence: list) -> list:
        """
        Method is utilized to eliminate stopwords from the sentence and making tokens lower case in case it is required.
        Note: source sentence will be handy when we eliminate features from the POS sentence
        :param source_sentence: sequence of the original tokens
        :param result_sentence: sequence of the original tokens (BIO) / sequence of POS features (POS)
        :return: cleaned sequence if cleaning is activated
        """
        clean_sentence = list()

        if self.configuration['stops']:
            for source_token, result_token in zip(source_sentence, result_sentence):
                if source_token['token'] not in self.configuration['stops']:
                    if self.configuration['task_name'] == 'POS':
                        clean_sentence.append({'token': result_token['token'], 'label': result_token['label']})
                    else:
                        clean_sentence.append(
                            {
                                'token': result_token['token'] if self.configuration['cased']
                                else result_token['token'].lower(),
                                'label': result_token['label']
                            }
                        )
        return clean_sentence

    def collect_sentences(self) -> collections.Iterable:
        """
        Method is utilized to yield sentences from the given dataset
        """
        sentence = list()
        sentence_delimiters = ['.', '?', '!', '__END_PARAGRAPH__', '__END_ESSAY__']
        for token in self.collect_tokens():
            sentence.append(token)
            if token['token'] in sentence_delimiters:
                result_sentence = sentence if self.configuration['task_name'] == 'BIO' else self.get_pos_tags(sentence)
                cleaned_sentence = self.clean_sentence(sentence, result_sentence)
                sentence = list()
                yield cleaned_sentence

    def collect_paragraphs(self) -> collections.Iterable:
        """
        Method is utilized to yield paragraphs from the given dataset
        """
        paragraph = list()
        for sentence in self.collect_sentences():
            paragraph.extend(sentence)
            if len(sentence) == 1:
                result_paragraph = paragraph
                paragraph = list()
                yield result_paragraph

    def collect_essays(self) -> collections.Iterable:
        """
        Method is utilized to yield essays from the given dataset
        """
        essay = list()
        for paragraph in self.collect_paragraphs():
            essay.extend(paragraph)
            if len(paragraph) == 1:
                result_essay = essay
                essay = list()
                yield result_essay

    @staticmethod
    def clean_data(data: list) -> tuple:
        """
        Method is utilized to return tuple of sequence of tokens and labels
        """
        tokens = list()
        labels = list()
        for each_data in data:
            tokens.append(each_data['token'])
            labels.append(each_data['label'])
        return tokens, labels

    def process_data(self) -> dict:
        """
        Method is utilized to combine all process in one
        :return: dictionary of required dataset
        """
        dataset = list()
        data_counter = 0

        if not os.path.exists(self.configuration['result_file']):
            for each in self.configuration[f'{self.configuration["process_type"]}_collect']:
                if len(each) == 1:
                    data_counter -= 1

                    dataset[data_counter].extend(each)
                else:
                    dataset.append(each)

                data_counter += 1
            data_dict = {'data': list(), 'labels': list()}
            ti = tqdm(dataset, total=len(dataset), desc=f'{self.configuration["process_type"].title()} is collected '
                                                        f'for {self.configuration["ds_type"]}'
                                                        f' of {self.configuration["task_name"]}')
            for each in ti:
                tokens, labels = self.clean_data(each)
                data_dict['data'].append(tokens)
                data_dict['labels'].append(labels)
            with open(self.configuration['result_file'], 'wb') as result_data:
                pickle.dump(data_dict, result_data)
        with open(self.configuration['result_file'], 'rb') as result_data:
            data_dict = pickle.load(result_data)
        return data_dict

    def __iter__(self) -> collections.Iterable:
        """
        Method makes the object perform as a generator
        """
        for each_data, each_labels in zip(self.dataset['data'], self.dataset['labels']):
            yield each_data, each_labels

    def __len__(self):
        """
        Method is utilized to compute the length of the dataset
        :return: number of the data that requested dataset contains
        """
        return len(self.dataset['data'])
