import argparse
import os.path
import nltk
nltk.download('stopwords')

import torch.cuda


def collect_arguments() -> argparse.Namespace:
    """
    Function is utilized to collect user-defined arguments to set project configuration
    :return: Namespace object which contains project parameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='dataset', required=False,
                        help='Specifies data directory')
    parser.add_argument('--process_type', type=str, default='paragraph', required=False, choices=['sentence', 'essay', 'paragraph'],
                        help='Specifies whether paragraph or essay will be considered for training')
    parser.add_argument('--window_size', type=int, default=200, required=False,
                        help='Specifies size of the window')
    parser.add_argument('--window_shift', type=int, default=200, required=False,
                        help='Specifies the size of the window shift')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='Specifies batch size for training and evaluation')
    parser.add_argument('--exp_num', type=int, default=13, required=False, help='Specifies the experiment number')
    parser.add_argument('--epochs', type=int, default=13, required=False,
                        help='Number of epochs that the model will be trained')
    parser.add_argument('--cased', default=False, action='store_true', required=False,
                        help='Specifies whether data will be upper/lower case or only lower case')
    parser.add_argument('--clean_stops', default=False, action='store_true', required=False,
                        help='Specifies whether stopwords will be cleaned or not')
    parser.add_argument('--split', default=0.0, type=float, required=False,
                        help='Specifies whether train set will be split into train/development sets or not')
    parser.add_argument('--embedding_dim', default=300, type=int, required=False,
                        help='Specfies embedding dimension of the model')
    parser.add_argument('--hidden_dim', default=200, type=int, required=False,
                        help='Specifies hidden dimensions of LSTM layers')
    parser.add_argument('--bidirectional', action='store_true', default=False, required=False,
                        help='Specifies whether Bidirectional LSTM (True) is used or not (False)')
    parser.add_argument('--num_lstm_layers', default=2, required=False, type=int,
                        help='Specifies number of lstm layers')
    parser.add_argument('--num_inputs', default=2, required=False, type=int,
                        help='Specifies whether multi-task approach is activated or not')
    parser.add_argument('--output_size', default=4, required=False, type=int,
                        help='Specifies number of outputs')
    parser.add_argument('--optimizer', default='SGD', required=False, type=str, choices=['Adam', 'SGD'],
                        help='Specifies optimizer type')
    parser.add_argument('--learning_rate', default=0.001, type=float, required=False,
                        help='Specifies learning rate of the model')
    parser.add_argument('--resume_training', default=False, action='store_true', required=False,
                        help= 'Specifies whether training will be continued or not')
    parser.add_argument('--num_epochs', default=150, type=int, required=False,
                        help='Specifies number of epochs which the modle will be trained')
    parser.add_argument('--init_eval', default=False, action='store_true', required=False,
                        help='Specifies whether initial evaluation is required (True) or not (False)')
    return parser.parse_args()


def collect_parameters() -> dict:
    """
    Function is utilized to collect user-specific parameters and add several parameters
    :return: dictionary that contains all required parameters for the project
    """
    arguments = collect_arguments()
    parameters = dict()
    for arg in vars(arguments):
        parameters[arg] = getattr(arguments, arg)

    parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameters['clean_stops'] = nltk.corpus.stopwords.words('english') if parameters['clean_stops'] else parameters['clean_stops']
    return parameters


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
