from train import TrainModel
from process_data import ProcessData
from collections import Counter, OrderedDict
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


class Statistics:
    def __init__(self, config_parameters):
        """
        
        :param config_parameters: 
        """
        self.configuration = self.set_configuration(config_parameters)
        self.id2label = self.revert_labels()

    def revert_labels(self):
        """
        Method is utilized to generate idx to label dictionary
        :return:
        """
        return {idx: label for label, idx in self.configuration['lab2id'].items()}

    @staticmethod
    def set_configuration(parameters):
        """

        :param parameters:
        :return:
        """
        trainer_obj = TrainModel(parameters)
        configuration = parameters

        configuration['train_object'] = trainer_obj
        configuration['environment'] = trainer_obj.configuration['environment']
        configuration['inference_dir'] = trainer_obj.configuration['inference_dir']
        configuration['vocabulary_bio'] = trainer_obj.configuration['vocabulary_bio']
        configuration['vocabulary_pos'] = trainer_obj.configuration['vocabulary_pos']
        configuration['split'] = parameters['split']
        configuration['lab2id'] = trainer_obj.configuration['lab2id']
        return configuration

    def vocabulary_info(self) -> None:
        """
        Method is utilized to provide vocabulary information for each task
        :return:
        """
        print(f'BIO task was set on vocabulary of {len(self.configuration["vocabulary_bio"])} elements')
        print(f'POS task was set on vocabulary of {len(self.configuration["vocabulary_pos"])} elements')

    def label_distribution(self) -> None:
        """
        Method is utilized to provide label distribution information of the datasets
        :return: None
        """
        ds_types = ['train', 'dev', 'test'] if self.configuration['split'] else ['train', 'test']
        for each_type in ds_types:
            labels = list()
            process_ = ProcessData(self.configuration, 'BIO', each_type)
            for each_label in process_.dataset['labels']:
                labels.extend(each_label)
            counter = Counter(labels)
            data = {k: v for k, v in counter.items()}
            print(f'Label distribution in {each_type} dataset is as follows: {data}')

    def provide_statistics(self, before: bool = True) -> None:
        """
        Method is utilized to provide statistics information for the project according to the provided argument
        :param before: specifies whether statistics which can be shown before (True) or after (False) the training
        :return: None
        """
        if before:
            self.vocabulary_info()
            self.label_distribution()

        else:
            self.plot_results(is_accuracy=True)
            self.plot_results(is_accuracy=False)
            self.get_confusion_details()

    def plot_results(self, is_accuracy: bool = True) -> None:
        """
        Method is utilized to plot train/dev results of the model according to the given parameter
        :param is_accuracy: boolean variable that specifies whether accuracy (True) or loss (False) results will be
                            plotted
        :return: None
        """
        results_dictionary_path = os.path.join(self.configuration['environment'], f'results.pickle')
        with open(results_dictionary_path, 'rb') as dict_path:
            results_dictionary = pickle.load(dict_path)

        ordered = OrderedDict(sorted(results_dictionary.items()))

        choice = 'accuracy' if is_accuracy else 'loss'
        dev_data = list()
        train_data = list()
        for epoch, results in ordered.items():
            dev_data.append(results[f'dev_{choice}'])
            train_data.append(results[f'train_{choice}'])

        plt.figure()
        plt.title(f'{choice.title()} results over {len(results_dictionary.keys())} epochs')
        plt.plot(list(results_dictionary.keys()), train_data, 'g', label='Train')
        plt.plot(list(results_dictionary.keys()), dev_data, 'r', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{choice.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(self.configuration['environment'], f'{choice}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def get_confusion_details(self, metric: str = 'f1_dev') -> None:
        """
        Method is utilized for inference over test set and creating confusion matrix
        :param metric: string object which specifies metric that the best model will be chosen accordingly
        :return: None
        """
        epoch = self.configuration['train_object'].load_model(metric=metric, is_best=True)
        inference_path = os.path.join(self.configuration['inference_dir'], f'inference_epoch_{epoch}.pickle')
        with open(inference_path, 'rb') as inf_data:
            inference_dict = pickle.load(inf_data)

        confusion = confusion_matrix(inference_dict['targets'], inference_dict['predictions'])
        print(confusion)
        self.plot_confusion_matrix(confusion)

    def plot_confusion_matrix(self, confusion: np.array) -> None:
        """
        Method is utilized to plot confusion matrix according to the given matrix
        :param confusion: numpy array for confusion matrix
        :return: None
        """
        labels = [self.id2label[idx] for idx in range(len(self.configuration['lab2id'])) if
                  self.id2label[idx] != '<PAD>']
        plt.figure(figsize=(8, 6), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(confusion, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)
        ax.set_title(f"Confusion Matrix", fontsize=14, pad=20)
        image_name = os.path.join(self.configuration['environment'], f'confusion_matrix.png')
        plt.savefig(image_name)
        plt.show()
