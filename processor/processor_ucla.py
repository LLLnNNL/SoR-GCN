from __future__ import print_function
import argparse
import os
import re
import time
import numpy as np
import yaml
import pickle
import math
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorboardX import SummaryWriter

plt.switch_backend('agg')


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


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.parameter_init(arg)
        self.load_model()
        self.load_weights()
        if self.arg.phase != 'test':
            self.load_optimizer()
        self.load_data()
        # self.A = [0]*60
        # self.B = [0]*60

    def parameter_init(self, arg):
        self.debug = arg.debug
        self.global_step = 0
        self.best_acc = False
        self.best_acc_value = 0

        now_time = re.sub(r'[ :]', r'-', time.asctime(time.localtime(time.time())))
        detail_data_path = os.path.join(arg.work_dir, 'runs', now_time)
        if not os.path.exists(detail_data_path):
            os.makedirs(detail_data_path)
        if arg.phase in ['train', 'hardpair_train', 'dcls_hardpairlearning_train', 'backbone_train']:
            self.train_writer = SummaryWriter(os.path.join(detail_data_path, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(detail_data_path, 'val'), 'val')
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        ration = str(self.arg.ob_ratio)
        if self.arg.phase not in ['2s', 'test']:
            dataset_num_train = len(self.arg.train_feeder_args['data_path'])
            self.data_loader['train'] = dict()
            for train_idx in range(dataset_num_train):
                self.data_loader['train'][ration] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args,
                                   num_class=self.arg.dcls_args['num_class'],
                                   debug=self.debug, data_type='train',
                                   data_random_ob=self.arg.data_random_ob,
                                   fill_type=self.arg.train_fill_type,
                                   device=self.output_device,
                                   times=self.arg.data_times,
                                   dataset=self.arg.dataset),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    drop_last=True,
                    worker_init_fn=init_seed)
        self.data_loader['test'] = dict()
        dataset_num_val = len(self.arg.test_feeder_args['data_path'])
        for val_idx in range(dataset_num_val):
            self.data_loader['test'][ration] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args,
                               num_class=self.arg.dcls_args['num_class'], debug=self.debug, data_type='test',
                               fill_type=self.arg.test_fill_type,
                               device=self.output_device,
                               dataset=self.arg.dataset),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)
    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # encoder
        Encoder = import_class(self.arg.model['encoder'])
        self.model_encoder = Encoder(**self.arg.encoder_args).cuda(output_device)
        model = self.model_encoder

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.print_log(f'# Parameters: {count_parameters(model)}')
        # D_cls
        Dcls = import_class(self.arg.model['dcls'])
        self.model_dcls = Dcls(**self.arg.dcls_args).cuda(output_device)
        # print(self.model_encoder)
        # print(self.model_dcls)

        # loss
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

    def load_weights(self):
        device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if self.arg.weights:
            weights = dict()
            for key in self.arg.weights:
                if '.pkl' in self.arg.weights[key]:
                    with open(self.arg.weights[key], 'r') as f:
                        weights[key] = pickle.load(f)
                else:
                    weights[key] = torch.load(self.arg.weights[key])

                weights[key] = OrderedDict([[k.split('module.')[-1], v.cuda(device)] for k, v in weights[key].items()])

                for w in self.arg.ignore_weights:
                    if weights[key].pop(w, None) is not None:
                        self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                    else:
                        self.print_log('Can Not Remove Weights: {}.'.format(w))
                if key == 'encoder':
                    model = self.model_encoder
                elif key == 'Dcls':
                    model = self.model_dcls


                try:
                    model.load_state_dict(weights[key])
                except:
                    state = model.state_dict()
                    diff = list(set(state.keys()).difference(set(weights[key].keys())))
                    print('Can not find these weights:')
                    for d in diff:
                        print('  ' + d)
                    state.update(weights[key])
                    model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model_dcls = nn.DataParallel(self.model_dcls, device_ids=self.arg.device,
                                                  output_device=device)

                self.model_encoder = nn.DataParallel(self.model_encoder, device_ids=self.arg.device,
                                                     output_device=device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer_E = optim.SGD(self.model_encoder.parameters(), lr=self.arg.base_lr, momentum=0.9,
                                         nesterov=self.arg.nesterov, weight_decay=self.arg.weight_decay)
            self.optimizer_Dcls = optim.SGD(self.model_dcls.parameters(), lr=self.arg.base_lr, momentum=0.9,
                                            nesterov=self.arg.nesterov, weight_decay=self.arg.weight_decay)

            self.lr_scheduler_E = ReduceLROnPlateau(self.optimizer_E, mode='min', factor=0.1,
                                                    patience=10, verbose=True,
                                                    threshold=1e-4, threshold_mode='rel',
                                                    cooldown=0)
            self.lr_scheduler_Dcls = ReduceLROnPlateau(self.optimizer_Dcls, mode='min', factor=0.1,
                                                       patience=10, verbose=True,
                                                       threshold=1e-4, threshold_mode='rel',
                                                       cooldown=0)
        else:
            self.print_log("The {} optimizer is not configured".format(self.arg.optimizer))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch, optimier, step=[30, 40], base_lr=0.0002, custom_lr=[], mode='common'):
        if mode == 'common':
            if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
                if epoch < self.arg.warm_up_epoch:
                    lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
                else:
                    #lr = base_lr / (10 ** np.sum(epoch >= np.array(step)))
                    # 台阶衰减（包含等号：到 step 当轮就降）
                    lr = self.arg.base_lr * (
                            self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))

                for param_group in optimier.param_groups:
                    param_group['lr'] = lr

                return lr
            else:
                raise ValueError()
        if mode == 'custom':
            if len(step) != len(custom_lr):
                raise ValueError('自定义参数长度不对称,step为{},custom_lr为{}'.format(len(step), len(custom_lr)))
            if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
                for (epoch_item, custom_lr_item) in zip(step, custom_lr):
                    if epoch == epoch_item:
                        lr = custom_lr_item
                        for param_group in optimier.param_groups:
                            param_group['lr'] = lr
                        return lr
            else:
                raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train_init(self):
        self.record_time()
        self.ED_preresultes = []
        self.Dcls_losses = []

    def Show_training_results(self):
        ED_acc = torch.mean(torch.cat(self.ED_preresultes).float()).item() * 100
        Dcls_mloss = np.mean(self.Dcls_losses)
        ED_Dfop_mloss = 0.0
        self.train_writer.add_scalar('ED_epoch_acc', ED_acc, self.epoch)
        self.train_writer.add_scalar('Dcls_epoch_mloss', Dcls_mloss, self.epoch)

        ED_mloss = Dcls_mloss + ED_Dfop_mloss
        self.print_log(
            '\t ED_epoch_acc: [{:.2f}% ] \tED_mloss: {:.4f} .'
                .format(ED_acc, ED_mloss))

    def change_LR(self):

        self.lr_E = self.adjust_learning_rate(self.epoch, self.optimizer_E, self.arg.step, self.arg.base_lr)
        self.lr_Dcls = self.adjust_learning_rate(self.epoch, self.optimizer_Dcls, self.arg.step, self.arg.base_lr)
        self.train_writer.add_scalar('lr_E', self.lr_E, self.global_step)
        self.train_writer.add_scalar('lr_Dcls', self.lr_Dcls, self.global_step)

        self.print_log('\t train_LR: \tEncoder[{:}] \tD_cls[{:}].'.format(self.lr_E, self.lr_Dcls))

    def save_model(self, model=None, name=''):
        save_path = self.arg.work_dir + '/model_w'
        os.makedirs(save_path, exist_ok=True)
        ob_ratio = self.arg.ob_ratio
        if model is None:
            state_dict_E = self.model_encoder.state_dict()
            state_dict_Dcls = self.model_dcls.state_dict()
            weights_E = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_E.items()])
            weights_Dcls = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_Dcls.items()])
            torch.save(weights_E, save_path + '/E-' + str(self.epoch + 1) + '.pt')
            torch.save(weights_Dcls, save_path + '/Dcls-' + str(self.epoch + 1) + '.pt')

            state_dict_Dfop = self.model_Dfop.state_dict()
            weights_Dfop = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_Dfop.items()])
            torch.save(weights_Dfop, save_path + '/Dfop-' + str(self.epoch + 1) + '.pt')
        else:
            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, save_path + '/' + name + '-' + str(ob_ratio) + '-' + str(self.epoch + 1) + '.pt')
            self.print_log('Save model_{}'.format(name))

        self.best_acc = False

    def freezeorunfreeze_model(self, model, a=False):
        for param in model.parameters():
            param.requires_grad = a


    def eval(self, epoch, save_score=False, loader_name='test'):
        # global A
        # global B
        self.model_encoder.eval()
        self.model_dcls.eval()
        self.print_log(
            'Eval epoch: {}\tdata:{}\ttest_fill_type:{}'.format(epoch + 1, loader_name, self.arg.test_fill_type))
        for key, loader in self.data_loader[loader_name].items():
            loss_value = []
            score_frag = []
            loader.dataset.set_OBR(self.arg.ob_ratio)
            # print(len(loader.dataset))
            for batch_idx, (data_F, data, label, index) in enumerate(tqdm(loader)):
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                with torch.no_grad():
                    output_E = self.model_encoder(data)
                    output_Dcls = self.model_dcls(output_E)


                loss = self.loss(output_Dcls, label)

                score_frag.append(output_Dcls.data.cpu().numpy())
                loss_value.append(loss.data.item())

                _, predict_label = torch.max(output_Dcls.data, 1)
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            accuracy_top1 = loader.dataset.top_k(score, 1) * 100
            accuracy_top5 = loader.dataset.top_k(score, 5) * 100

            if accuracy_top1 > self.best_acc_value:
                self.best_acc_value = accuracy_top1
                self.best_acc = True

            self.val_writer.add_scalar('epoch_TOP1', accuracy_top1, self.epoch)
            self.val_writer.add_scalar('epoch_TOP5', accuracy_top5, self.epoch)
            self.val_writer.add_scalar('epoch_loss', loss, self.epoch)

            self.print_log('\tOBR{}: Mean_loss[{:.4f}] \t Acc[Top1:{:.2f}% Top5:{:.2f}%].'
                           .format(self.arg.ob_ratio, loss, accuracy_top1, accuracy_top5))

    def test_eval(self, epoch, save_score=False, loader_name='test'):
        self.model_encoder.eval()
        self.model_dcls.eval()
        # print(self.A, self.B)

        self.print_log(
            'Eval epoch: {}\tdata:{}\ttest_fill_type:{}'.format(epoch + 1, loader_name, self.arg.test_fill_type))
        OBR_list = [100, 80, 60, 40, 20]
        for  a, loader in self.data_loader[loader_name].items():
            for key in OBR_list:
                loss_value = []
                score_frag = []

                loader.dataset.set_OBR(key)
                # print(len(loader.dataset))
                for batch_idx, (data_F, data, label, index) in enumerate(tqdm(loader)):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    with torch.no_grad():
                        output_E = self.model_encoder(data)
                        output_Dcls = self.model_dcls(output_E)

                    loss = self.loss(output_Dcls, label)
                    predict = output_Dcls.argmax(dim=-1).tolist()
                    truth = label.tolist()

                    score_frag.append(output_Dcls.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output_Dcls.data, 1)
                score = np.concatenate(score_frag)
                loss = np.mean(loss_value)

                accuracy_top1 = loader.dataset.top_k(score, 1) * 100
                accuracy_top5 = loader.dataset.top_k(score, 5) * 100

                self.val_writer.add_scalar('epoch_TOP1', accuracy_top1, key)
                self.val_writer.add_scalar('epoch_TOP5', accuracy_top5, key)
                self.val_writer.add_scalar('epoch_loss', loss, key)

                self.print_log('\tOBR{}: Mean_loss[{:.4f}] \t Acc[Top1:{:.2f}% Top5:{:.2f}%].'
                               .format(key, loss, accuracy_top1, accuracy_top5))

    def train(self):
        self.model_encoder.train()
        self.model_dcls.train()

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        self.train_init()

        ob_ratio = str(self.arg.ob_ratio)
        loader = self.data_loader['train'][ob_ratio]
        loader.dataset.res_random_tag()
        loader.dataset.set_OBR(self.arg.ob_ratio)


        for batch_idx, (data_F, data, label, index) in enumerate(tqdm(loader)):
            self.global_step += 1
            if isinstance(label, tuple):
                print('lable is tuple',label)
            with torch.no_grad():
                data_F = data_F.float().cuda(self.output_device)
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            ## Encoder&Dcls Training
            output_E_P = self.model_encoder(data)
            output_Dcls_P = self.model_dcls(output_E_P)

            Dcls_loss = self.loss(output_Dcls_P, label)
            ED_loss = Dcls_loss

            self.optimizer_E.zero_grad()
            self.optimizer_Dcls.zero_grad()
            ED_loss.backward()
            self.optimizer_E.step()
            self.optimizer_Dcls.step()

            timer['model'] += self.split_time()

            value, predict_label = torch.max(output_Dcls_P.data, 1)

            self.Dcls_losses.append(Dcls_loss.item())
            self.ED_preresultes.append(predict_label == label)

            timer['statistics'] += self.split_time()

        self.Show_training_results()

    def start_merge(self):
        self.print_log('Phase:{}\t OBR:{}\t Backbone:{} \tdata_random_ob:{}'
                       .format(self.arg.phase, self.arg.ob_ratio, self.arg.backbone, self.arg.data_random_ob))
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.epoch = epoch
                self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
                self.print_log(
                    'Training epoch: {}\ttrain_fill_type:{}'.format(self.epoch + 1, self.arg.train_fill_type))
                self.change_LR()
                self.train()

                # 验证
                if (epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 >= self.arg.Detailed_test_epoch:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name='test')
                if (epoch + 1) == self.arg.num_epoch or (self.best_acc and epoch + 1 >= self.arg.Detailed_test_epoch):
                    self.test_eval(0, save_score=self.arg.save_score, loader_name='test')
                # 保存模型
                if self.best_acc and self.best_acc and epoch + 1 >= self.arg.Detailed_test_epoch:
                    self.save_model(self.model_encoder, 'E-Best')
                    self.save_model(self.model_dcls, 'Dcls-Best')
                if ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.save_model(self.model_encoder, 'E')
                    self.save_model(self.model_dcls, 'Dcls')

                self.best_acc = False
            return self.best_acc_value
        elif self.arg.phase == 'test':
            self.test_eval(0, save_score=self.arg.save_score, loader_name='test')
