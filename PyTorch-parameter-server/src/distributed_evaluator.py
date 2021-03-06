from __future__ import print_function
import os.path
import time
import argparse
from datetime import datetime
import copy

import numpy as np

from nn_ops import NN_Trainer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_ops.lenet import LeNet
from model_ops.resnet import *
from util import build_model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Validation settings
    parser.add_argument('--eval-batch-size', type=int, default=10000, metavar='N',
                        help='the batch size when doing model validation, complete at once on default')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--model-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    args = parser.parse_args()
    return args

class DistributedEvaluator(NN_Trainer):
    '''
    The DistributedEvaluator aims at providing a seperate node in the distributed cluster to evaluate
    the model on validation/test set and return the results
    In this version, the DistributedEvaluator will only load the model from the dir where the master
    save the model and do the evaluation task based on a user defined frequency 
    '''
    def __init__(self, **kwargs):
        self._cur_step = 0
        self._model_dir = kwargs['model_dir']
        self._eval_freq = int(kwargs['eval_freq'])
        self._eval_batch_size = kwargs['eval_batch_size']
        self.network_config = kwargs['network']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def evaluate(self, validation_loader):
        # init objective to fetch at the begining
        self._next_step_to_fetch = self._cur_step + self._eval_freq
        self._num_batch_per_epoch = len(validation_loader) / self._eval_batch_size
        # check if next temp model exsits, if not we wait here else we continue to do the model evaluation
        while True:
            model_dir_=self._model_dir_generator(self._next_step_to_fetch)
            print(model_dir_)
            if os.path.isfile(model_dir_):
                self._load_model(model_dir_)
                print("Evaluator evaluating results on step {}".format(self._next_step_to_fetch))
                self._evaluate_model(validation_loader)
                self._next_step_to_fetch += self._eval_freq
            else:
                # TODO(hwang): sleep appropriate period of time make sure to tune this parameter
                time.sleep(10)

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        test_losses = []
        with torch.no_grad():
          for data, target in test_loader:
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    def _load_model(self, file_path):
        self.network = build_model(self.network_config, num_classes=10)
        with open(file_path, "rb") as f_:
            self.network.load_state_dict(torch.load(f_))

    def _model_dir_generator(self, next_step_to_fetch):
        return self._model_dir+"model_step_"

if __name__ == "__main__":
    # this is only a simple test case
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch Distributed Evaluator'))

    # load training and test set here:
    if args.dataset == "MNIST":
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.eval_batch_size, shuffle=True)
    elif args.dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                 shuffle=False)
    
    kwargs_evaluator={
                    'network':args.network,
                    'model_dir':args.model_dir, 
                    'eval_freq':args.eval_freq, 
                    'eval_batch_size':args.eval_batch_size}
    evaluator_nn = DistributedEvaluator(**kwargs_evaluator)
    evaluator_nn.evaluate(validation_loader=test_loader)
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))