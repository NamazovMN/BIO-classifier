import torch.nn as nn
import torch
import torch.nn.functional as f


class BioClassifier(nn.Module):
    """
    Class is used as an object for the classification model
    """

    def __init__(self, config_parameters: dict):
        """
        Method is an initializer of the class
        :param config_parameters: required parameters for the project
        """
        super(BioClassifier, self).__init__()
        self.max_length = config_parameters['window_size']
        self.embedding_pos = nn.Embedding(len(config_parameters['vocabulary_pos']), config_parameters['embedding_dim'],
                                          padding_idx=config_parameters['vocabulary_pos']['<PAD>'])
        self.embedding_bio = nn.Embedding(len(config_parameters['vocabulary_bio']), config_parameters['embedding_dim'],
                                          padding_idx=config_parameters['vocabulary_bio']['<PAD>'])
        self.LSTM_bio = self.set_lstm(config_parameters)
        self.LSTM_pos = self.set_lstm(config_parameters)
        lstm_out = config_parameters['hidden_dim'] * 2 if config_parameters['bidirectional'] else config_parameters[
            'hidden_dim']
        self.linear1 = nn.Linear(config_parameters['num_inputs'] * lstm_out, 148)
        self.linear2 = nn.Linear(148, config_parameters['output_size'])
        self.relu = nn.ReLU()

    @staticmethod
    def set_lstm(parameters: dict) -> nn.LSTM:
        """
        Method is utilized to set the LSTM layer for each task
        :param parameters: required parameters for the project
        :return: LSTM layer for each task
        """
        return nn.LSTM(parameters['embedding_dim'], parameters['hidden_dim'], bidirectional=parameters['bidirectional'],
                       num_layers=parameters['num_lstm_layers'], batch_first=True)

    def forward(self, input_data: dict) -> torch.Tensor:
        """
        Method is utilized to perform feedforward process of the model
        :param input_data: dictionary which contains data loaders for each task
        :return: output of the classifier
        """
        x_pos = input_data['pos']
        x_bio = input_data['bio']

        out_emb_bio = self.embedding_bio(x_bio)
        out_emb_pos = self.embedding_pos(x_pos)

        lstm_bio, (_, _) = self.LSTM_bio(out_emb_bio)
        lstm_pos, (_, _) = self.LSTM_pos(out_emb_pos)

        lstm_bio = self.relu(lstm_bio)
        lstm_pos = self.relu(lstm_pos)

        in_linear = torch.cat((lstm_bio, lstm_pos), -1)
        output1 = self.linear1(in_linear)
        output2 = self.linear2(self.relu(output1))
        output = f.log_softmax(output2, dim=1)

        return output
