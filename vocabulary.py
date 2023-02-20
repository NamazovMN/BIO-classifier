import pickle
import os


class Vocabulary:
    """
    Class sets the vocabularies for each task
    """
    def __init__(self, config_parameters: dict):
        """
        Method is utilized as initializer of the class
        :param config_parameters: required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.vocabulary, self.lab2id = self.get_vocabulary() if self.configuration[
                                                                    'task_name'] == 'BIO' else self.get_pos_vocabulary()

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method is utilized to set the configuration of the specific task
        :param parameters: required parameters for the project
        :return: parameters that will be used for this specific task
        """
        return {
            'task_name': parameters['task_name'],
            'source_path': parameters['result_file'],
            'vocab_dir': os.path.join(parameters['output_dir'],
                                      f'vocabulary_{parameters["task_name"]}_{parameters["process_type"]}.pickle'),
            'lab2id_dir': os.path.join(parameters['output_dir'],
                                       f'lab2id_{parameters["process_type"]}.pickle')
        }

    def get_vocabulary(self) -> tuple:
        """
        Method is utilized to get vocabulary for the BIO task
        :return: tuple of vocabulary and dictionary of labels
        """
        if not os.path.exists(self.configuration["vocab_dir"]):
            with open(self.configuration['source_path'], 'rb') as source_dir:
                source_data = pickle.load(source_dir)
            tokens = list()
            targets = list()
            for data, labels in zip(source_data['train']['data'], source_data['train']['labels']):
                tokens.extend(data)
                targets.extend(labels)
            vocabulary = {token: idx for idx, token in enumerate(set(tokens))}
            vocabulary['<UNK>'] = len(vocabulary)
            vocabulary['<PAD>'] = len(vocabulary)
            lab2id = {label: idx for idx, label in enumerate(set(targets))}
            lab2id['<PAD>'] = len(lab2id)
            with open(self.configuration['vocab_dir'], 'wb') as vocab_path:
                pickle.dump(vocabulary, vocab_path)
            with open(self.configuration['lab2id_dir'], 'wb') as lab2id_path:
                pickle.dump(lab2id, lab2id_path)
        with open(self.configuration['vocab_dir'], 'rb') as vocab_path:
            vocabulary = pickle.load(vocab_path)
        with open(self.configuration['lab2id_dir'], 'rb') as lab2id_path:
            lab2id = pickle.load(lab2id_path)

        return vocabulary, lab2id

    def get_pos_vocabulary(self) -> tuple:
        """
        Method is utilized to collect vocabulary information for the pos tagged dataset
        :return: tuple of vocabulary and dictionary of labels
        """
        if not os.path.exists(self.configuration['lab2id_dir']):
            raise FileNotFoundError('Vocabulary object must be created on BIO dataset, but you are trying to create it'
                                    'with POS dataset!')
        if not os.path.exists(self.configuration['vocab_dir']):
            with open(self.configuration['source_path'], 'rb') as source_dir:
                source_data = pickle.load(source_dir)
            tokens = list()
            for data in source_data['train']['data']:
                tokens.extend(data)
            vocabulary = {token: idx for idx, token in enumerate(set(tokens))}
            vocabulary['<UNK>'] = len(vocabulary)
            vocabulary['<PAD>'] = len(vocabulary)
            with open(self.configuration['vocab_dir'], 'wb') as vocab_data:
                pickle.dump(vocabulary, vocab_data)
        with open(self.configuration['vocab_dir'], 'rb') as vocab_data:
            vocabulary = pickle.load(vocab_data)
        with open(self.configuration['lab2id_dir'], 'rb') as lab2id_data:
            lab2id = pickle.load(lab2id_data)

        return vocabulary, lab2id
    def __getitem__(self, item: str) -> int:
        """
        Method is utilized as a getter for the vocabulary. In case element is not in vocabulary, index oof <UNK> will
        be set to the token
        :param item: word or pos tag which is requested to be encoded by vocabulary
        :return: index of the provided item
        """
        return self.vocabulary[item] if item in self.vocabulary.keys() else self.vocabulary['<UNK>']

    def __len__(self) -> int:
        """
        Method is utilized to provide the length of the vocabulary data
        :return: number of elements in the vocabulary
        """
        return len(self.vocabulary)
