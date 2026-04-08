#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import os
import time
import numpy as np
import yaml
# torch
import torch
import torch.backends.cudnn as cudnn


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/nturgbd-cross-subject/Random_OBR_20.yaml',
                        help='path to the configuration file')
    parser.add_argument('--debug', default=True,help='debug processor')
    parser.add_argument('--ob-ratio',type=int, default=100,help='Observation ratio')
    parser.add_argument('--save-score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # processor
    parser.add_argument('--processor', default='processor.processor', help='Execute training and validation procedures')
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=10, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
    parser.add_argument('--Detailed_test_epoch', type=int, default=35,
                        help='When epoch >Detailed_test_epoch, each epoch is tested after training')

    # dataset
    parser.add_argument('--dataset', default='NTU', help='training dataset')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=0, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--data_frame', default=100, help='input data frame')
    parser.add_argument('--data_times', default=1, help='data repeat times')

    # model
    parser.add_argument('--model', default=dict(), help='the model will be used')
    parser.add_argument('--encoder-args', type=dict, default=dict(), help='the arguments of encoder')
    parser.add_argument('--dcls-args', type=dict, default=dict(), help='the arguments of dcls')
    parser.add_argument('--dfop-args', type=dict, default=dict(), help='the arguments of dfop')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--lr-decay-rate',type=float,default=0.1,help='decay rate for learning rate')
    #train
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=65, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--data_random_ob', default=False)
    parser.add_argument('--backbone', default=True)
    parser.add_argument('--Lable_encoding', default=False,help='empty/single/double')

    #data processing
    parser.add_argument('--train_fill_type', default='repeat', help='zero/linear/repeat')
    parser.add_argument('--test_fill_type', default='repeat',help='zero/linear/repeat')

    return parser


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_arg(p):
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    init_seed(0)

    return arg


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    #p = parser.parse_args(['--config', './config/nturgbd-cross-subject/backbone_train_joint.yaml'])
    p = parser.parse_args()

    arg = get_arg(p)

    Processor = import_class(arg.processor)
    processor = Processor(arg)
    if processor.arg.phase in ['train','test']:
        processor.start_merge()
    elif processor.arg.phase == '2s':
        processor.start()
