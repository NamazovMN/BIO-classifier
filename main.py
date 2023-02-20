from src.train import TrainModel
from statistics import Statistics
from train import TrainModel
from utilities import *


def __main__():
    parameters = collect_parameters()
    statistics = Statistics(parameters)
    statistics.provide_statistics()
    runner = TrainModel(parameters)
    runner.train_epoch()
    statistics.provide_statistics(False)


if __name__ == '__main__':
    __main__()
