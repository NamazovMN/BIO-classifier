import os
import pickle

import torch
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ArgumentDataset
from model import BioClassifier
from process_data import ProcessData
from utilities import *


class TrainModel:
    def __init__(self, config_parameters: dict):
        """
        Method performs as an initializer of the class
        :param config_parameters: configuration parameters of the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.model = self.set_model()
        self.loss = self.set_loss_fn()
        self.optimizer = self.set_optimizer(config_parameters)

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method is utilized to extract task-specific parameters from the configuration parameters of the project
        :param parameters: configuration parameters of the project
        :return: dictionary that contains task-specific parameters
        """
        output_dir = 'train_results'
        check_dir(output_dir)
        experiment_environment = os.path.join(output_dir, f'experiment_{parameters["exp_num"]}')
        check_dir(experiment_environment)

        self.save_parameters(experiment_environment, parameters)
        checkpoints_directory = os.path.join(experiment_environment, 'checkpoints')
        check_dir(checkpoints_directory)
        inference_directory = os.path.join(experiment_environment, 'inferences')
        check_dir(inference_directory)
        process_bio = ProcessData(parameters, 'BIO', 'test')
        process_pos = ProcessData(parameters, 'POS', 'test')
        configuration = parameters
        configuration['environment'] = experiment_environment

        configuration['ckpt_dir'] = checkpoints_directory
        configuration['inference_dir'] = inference_directory
        configuration['bio_loaders'] = self.get_loaders(parameters, "BIO")
        configuration['pos_loaders'] = self.get_loaders(parameters, 'POS')
        configuration['vocabulary_bio'] = process_bio.vocabulary
        configuration['vocabulary_pos'] = process_pos.vocabulary
        configuration['lab2id'] = process_pos.lab2id
        return configuration

    def set_loss_fn(self) -> CrossEntropyLoss:
        """
        Method is used to set loss function according to the provided parameters
        :return: Loss function
        """
        return CrossEntropyLoss(ignore_index=self.configuration['lab2id']['<PAD>'])

    def set_optimizer(self, parameters: dict) -> Adam:
        """
        Method is used to set optimizer according to the provided parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: Optimizer for the model
        """
        if parameters['optimizer'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(), lr=parameters['learning_rate'])
        elif parameters['optimizer'] == 'SGD':
            optimizer = SGD(params=self.model.parameters(), lr=parameters['learning_rate'], momentum=0.8)
        else:
            raise Exception('There is not such optimizer in our scenarios. You should choose one of SGD or Adam')
        return optimizer

    @staticmethod
    def save_parameters(output_dir: str, parameters_dict: dict) -> None:
        """
        Method is used to save configuration dictionary to the given path
        :param output_dir: output path where model configuration parameters will be saved
        :param parameters_dict: configuration parameters include all relevant data for the process
        :return: None
        """
        file_name = os.path.join(output_dir, 'model_config.pickle')
        with open(file_name, 'wb') as out_dir:
            pickle.dump(parameters_dict, out_dir)

    def set_model(self) -> BioClassifier:
        """
        Method is utilized to set the classifier model
        :return: BIO classification model for the project
        """
        model = BioClassifier(self.configuration).to(self.configuration['device'])
        return model

    @staticmethod
    def get_loaders(parameters: dict, task_name: str):
        """
        Method is utilized to collect task-specific data loaders
        :param parameters:
        :param task_name:
        :return:
        """

        types = ['train', 'dev', 'test'] if parameters['split'] else ['train', 'test']

        loaders_dict = {each_type: DataLoader(ArgumentDataset(parameters, task_name, each_type),
                                              batch_size=parameters['batch_size'], shuffle=True) for each_type in types}
        return loaders_dict

    def compute_acc(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Method is used to compute accuracy according to the provided ground truth and prediction values
        :param prediction: output of the model
        :param target: ground truth labels
        :return: tuple which contains following information:
                correct: number of correct predictions in the provided batch
                length_data: number of data were used to compute the accuracy
                original_predictions: list of predictions, in which tokens from <PAD> positions are discarded
                original_targets: list of targets, in which <PAD> tokens are discarded
        """
        pred_list = torch.argmax(prediction, dim=1).tolist()

        target_list = target.view(-1).tolist()

        original_predictions = list()
        original_targets = list()
        accuracy_list = list()

        for pred, label in zip(pred_list, target_list):
            if label != self.configuration['lab2id']['<PAD>']:
                original_predictions.append(pred)
                original_targets.append(label)
                accuracy_list.append(pred == label)

        correct = sum(accuracy_list)
        length_data = len(original_targets)
        return correct, length_data, original_predictions, original_targets

    def train_step(self, bio_loader: dict, pos_loader: dict) -> tuple:
        """
        Method is used to perform training for one step (or for one batch)
        :param bio_loader: dictionary which includes train data tensor and label tensor for bio task
        :param pos_loader: dictionary which includes train data tensor and label tensor for pos task
        :return: tuple which contains following information:
                loss.item(): training loss for this specific batch
                accuracy: accuracy value for this very batch training
                num_tokens: number of tokens that batch includes
        """
        self.optimizer.zero_grad()
        output = self.model({'bio': bio_loader['data'].to(self.configuration['device']),
                             'pos': pos_loader['data'].to(self.configuration['device'])})
        output = output.view(-1, output.shape[-1])
        labels = bio_loader['label'].view(-1)

        loss = self.loss(output, labels.to(self.configuration['device']))

        loss.backward()
        self.optimizer.step()
        accuracy, num_tokens, _, _ = self.compute_acc(output, bio_loader['label'])
        return loss.item(), accuracy, num_tokens

    def train_epoch(self) -> None:
        """
        Method is used to perform training for epochs
        :return: None
        """

        chosen_epoch = 0

        if self.configuration['resume_training']:
            print(
                'Please, be sure that model satisfies saved parameters in model_parameters.config in experiment folder')
            chosen_epoch = self.load_model(is_best=False)

        train_range = range(self.configuration['num_epochs']) if not chosen_epoch else range(chosen_epoch + 1,
                                                                                             self.configuration[
                                                                                                 'num_epochs'])

        train_loaders = {'pos': self.configuration['pos_loaders']['train'],
                         'bio': self.configuration['bio_loaders']['train']}

        num_batches = len(train_loaders['bio'])
        if self.configuration['init_eval']:
            self.evaluate_epoch(epoch=chosen_epoch)

        for epoch in train_range:
            print(f'{20 * "<<"} EPOCH {epoch} {20 * ">>"}')
            epoch_loss = 0
            accuracy = 0

            ti = tqdm(iterable=zip(train_loaders['pos'], train_loaders['bio']), total=num_batches, leave=True)
            total_tokens = 0
            epoch_accuracy = 0
            for batch_pos, batch_bio in ti:
                self.model.train()
                step_loss, step_accuracy, num_tokens = self.train_step(batch_bio, batch_pos)

                total_tokens += num_tokens
                epoch_accuracy += step_accuracy
                epoch_loss += step_loss
                accuracy += step_accuracy

                ti.set_description(f'Epoch: {epoch}, TRAIN -> epoch loss: {epoch_loss / num_batches : .4f}, '
                                   f'accuracy : {epoch_accuracy / total_tokens : .4f}')

            dev_loss, dev_accuracy, num_batches_dev, total_tokens_dev, f1_dev = self.evaluate_epoch(epoch)
            epoch_dict = {
                'train_loss': epoch_loss / num_batches,
                'train_accuracy': accuracy / total_tokens,
                'dev_loss': dev_loss / num_batches_dev,
                'dev_accuracy': dev_accuracy / total_tokens_dev,
                'f1_dev': f1_dev,
                # 'f1_baseline': baseline_f1
            }
            print(f'F1 score for evaluation data: {f1_dev}')

            self.save_results(epoch_dict, epoch)

    def evaluate_epoch(self, epoch: int) -> tuple:
        """
        Method is used to evaluate the model on dev data after each epoch
        :param epoch: current epoch number
        :return: tuple that contains following information:
                dev_loss: validation loss for this epoch
                dev_accuracy: validation accuracy for this epoch
                num_batches: number of batches in validation set
                total_tokens: number of tokens in validation loader
                f1: f1 score for current epoch, that is computed on dev dataset
        """
        choice = 'dev' if self.configuration['split'] else 'test'

        dev_loaders = {
            'pos': self.configuration['pos_loaders'][choice],
            'bio': self.configuration['bio_loaders'][choice]
        }
        num_batches = len(dev_loaders['bio'])
        dev_loss = 0
        dev_accuracy = 0
        total_tokens = 0
        ti = tqdm(iterable=zip(dev_loaders['bio'], dev_loaders['pos']), total=num_batches,
                  desc=f'Epoch: {epoch}, VALIDATION -> epoch loss: {dev_loss}, accuracy : {dev_accuracy}')

        self.model.eval()
        targets = list()
        predictions = list()
        for batch_bio, batch_pos in ti:
            output = self.model({'bio': batch_bio['data'].to(self.configuration['device']),
                                 'pos': batch_pos['data'].to(self.configuration['device'])})
            output = output.view(-1, output.shape[-1])
            labels = batch_bio['label'].view(-1)
            loss = self.loss(output, labels.to(self.configuration['device']))
            acc_dev_step, num_tokens, prediction_sentences, target_sentences = self.compute_acc(output,
                                                                                                batch_bio['label'])
            dev_accuracy += acc_dev_step
            dev_loss += loss.item()
            total_tokens += num_tokens
            ti.set_description(f'Epoch: {epoch}, VALIDATION -> epoch loss: {dev_loss / num_batches: .4f}, '
                               f'accuracy : {dev_accuracy / total_tokens :.4f}')
            targets.extend(target_sentences)
            predictions.extend(prediction_sentences)

        f1 = f1_score(targets, predictions, average='macro')
        file_name = os.path.join(self.configuration['inference_dir'], f'inference_epoch_{epoch}.pickle')
        inference_dict = {
            'targets': targets,
            'predictions': predictions
        }
        with open(file_name, 'wb') as inf_data:
            pickle.dump(inference_dict, inf_data)

        return dev_loss, dev_accuracy, num_batches, total_tokens, f1

    def get_epoch(self, is_best: bool = False, metric: str = 'f1_dev') -> int:
        """
        Method is utilized to get the epoch which is chosen according to the provided arguments
        :param is_best: boolean variable specifies whether the best (True) or the last model (False) is required
        :param metric: metric will be used to select the best epoch according to that
        :return: chosen epoch
        """
        with open(os.path.join(self.configuration['environment'], 'results.pickle'), 'rb') as res_file:
            train_results = pickle.load(res_file)
        results = {epoch: data[metric] for epoch, data in train_results.items()}
        if is_best:
            epoch = max(results, key=results.get)
        else:
            epoch = max(results.keys())
        return epoch

    def save_results(self, result_dict: dict, epoch: int) -> None:
        """
        Method is utilized for saving training results and model parameters after training the specific epoch
        :param result_dict: dictionary contains training results for epoch
        :param epoch: integer specifies current epoch
        :return: None
        """
        file_name = os.path.join(self.configuration['environment'], 'results.pickle')
        result = dict()
        if not os.path.exists(file_name):
            result[epoch] = result_dict
        else:
            with open(file_name, 'rb') as data:
                result = pickle.load(data)
            result[epoch] = result_dict
        with open(file_name, 'wb') as data:
            pickle.dump(result, data)

        model_dict_name = os.path.join(self.configuration['ckpt_dir'],
                                       f"model_{epoch}_f1_{result_dict['f1_dev']: .3f}_"
                                       f"dl_{result_dict['dev_loss']: .3f}_"
                                       f"tl_{result_dict['train_loss']: .3f}_"
                                       f"da_{result_dict['dev_accuracy']: .3f}")
        optimizer_dict_name = os.path.join(self.configuration['ckpt_dir'], f'optim_epoch_{epoch}')
        torch.save(self.model.state_dict(), model_dict_name)
        torch.save(self.optimizer.state_dict(), optimizer_dict_name)

    def load_model(self, metric: str = 'f1_dev', is_best: bool = False) -> int:
        """
        Method is used to load the model according to is_best value
        :param metric: specifies the metric which will be utilized to get the best model
        :param is_best: boolean variable specifies the best one is requested or not
        :return: None
        """
        epoch = self.get_epoch(is_best, metric)
        file_names = {'model_path': str(), 'optim_path': str()}
        print(epoch)
        for each in os.listdir(self.configuration['ckpt_dir']):
            print(each)
            if f'model_{epoch}_' in each:
                file_names['model_path'] = os.path.join(self.configuration['ckpt_dir'], each)
                file_names['optim_path'] = os.path.join(self.configuration['ckpt_dir'], f'optim_epoch_{epoch}')
                break
        if not file_names['model_path']:
            raise FileNotFoundError('No path was found, you need to train the model first!')

        if file_names['model_path']:
            print(f"Model is loaded from {file_names['model_path']}")
            self.model.load_state_dict(torch.load(file_names['model_path'], map_location=self.configuration['device']))
            self.optimizer.load_state_dict(
                torch.load(file_names['optim_path'], map_location=self.configuration['device']))
            self.model.eval()
        else:
            raise FileNotFoundError('No trained model was found')

        return epoch
