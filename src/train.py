from torch.utils.data import DataLoader
from utilities import *
from src.dataset import BIO_Dataset
class TrainModel:
    def __init__(self, config_parameters):
        self.configuration = self.set_configuration(config_parameters)

    def set_configuration(self, parameters):
        train_results = 'train_results'
        experiment_environment = os.path.join(train_results, f"experiment_{parameters['exp_num']}")
        check_dir(experiment_environment)
        inferences = os.path.join(experiment_environment, 'inferences')
        check_dir(inferences)
        checkpoints = os.path.join(experiment_environment, 'checkpoints')

        return {
            'epochs': parameters['epochs'],
            'output_dir': experiment_environment,
            'inferences': inferences,
            'chkpt_dir': checkpoints,
            'bio_loaders': self.set_loaders(parameters, 'BIO'),
            'pos_loaders': self.set_loaders(parameters, 'POS')
        }

    def set_loaders(self, parameters, task_name):
        loaders_dict = {f'{ds_type}_loader': DataLoader(
            BIO_Dataset(parameters, ds_type, task_name),
            batch_size=parameters['batch_size'],
            shuffle=True
        ) for ds_type in ['train', 'test']}
        return loaders_dict
    def train_model(self):
        bio_loaders = self.configuration['bio_loaders']
        print(bio_loaders)
        pos_loaders = self.configuration['pos_loaders']
        for each in bio_loaders['train']:
            print(each['data'])
            print(each['label'])
            input()


