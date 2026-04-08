import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
from feeders import tools
from torch.nn import functional as F
import math
from .bone_pairs import ntu_pairs, kinect_leg_pairs, kinect_hand_pairs,FPHA_pairs,UCLA_pairs
from .tools import sample_yaw, rotate_xyz, leftright_flip, scale_translate_xz,time_resample_keep_len, per_sequence_scale_norm, ntu_lr_pairs

sys.path.extend(['../'])


def get_train_clips(num_frames, clip_len, num_clips, p_interval, float_ok, seed):
    """Uniformly sample indices for training clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    allinds = []
    for clip_idx in range(num_clips):
        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if float_ok:
            interval = (num_frames - 1) / clip_len
            offsets = np.arange(clip_len) * interval
            inds = np.random.rand(clip_len) * interval + offsets
            inds = inds.astype(np.float32)
        elif num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + off
        num_frames = old_num_frames

        allinds.append(inds)

    return np.concatenate(allinds)


def get_test_clips(num_frames, clip_len, num_clips, p_interval, float_ok, seed):
    """Uniformly sample indices for testing clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    np.random.seed(seed)
    if float_ok:
        interval = (num_frames - 1) / clip_len
        offsets = np.arange(clip_len) * interval
        inds = np.concatenate([
            np.random.rand(clip_len) * interval + offsets
            for i in range(num_clips)
        ]).astype(np.float32)

    all_inds = []

    for i in range(num_clips):

        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start_ind = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start_ind, start_ind + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        all_inds.append(inds + off)
        num_frames = old_num_frames

    return np.concatenate(all_inds)


def CropFrame(data, tar_len=100, test_mode=False, num_clips=1, p_interval=(1, 1), float_ok=False, seed=255,device=0):
    num_frames = data.shape[1]
    if test_mode:
        inds = get_test_clips(num_frames, tar_len, num_clips, p_interval, float_ok, seed)
    else:
        inds = get_train_clips(num_frames, tar_len, num_clips, p_interval, float_ok, seed)

    inds = np.mod(inds, num_frames)

    kp = data.copy()
    # 记录每一帧的人数
    num_person = kp.shape[0]
    num_persons = [num_person] * num_frames
    for i in range(num_frames):
        j = num_person
        while j >= 0 and np.all(np.abs(kp[j - 1, i]) < 1e-5):
            j -= 1
        num_persons[i] = j

    # 找到帧与帧之间人数有变化的位置
    transitional = [False] * num_frames
    for i in range(1, num_frames - 1):
        if num_persons[i] != num_persons[i - 1]:
            transitional[i] = transitional[i - 1] = True
        if num_persons[i] != num_persons[i + 1]:
            transitional[i] = transitional[i + 1] = True

    inds_int = inds.astype(np.int)
    coeff = np.array([transitional[i] for i in inds_int])
    inds = (coeff * inds_int + (1 - coeff) * inds)

    out_results = np.array([data[:, i, :, :] for i in inds]).transpose([1, 0, 2, 3])
    return out_results[:, :tar_len, :, :]


def FillFrame(data, tar_len=100, device=0):
    num_person, num_frames, V, C = data.shape

    # 记录每一帧的人数
    num_persons = np.full(num_frames, 2, dtype=np.int64)
    for i in range(num_frames):
        j = num_person
        while j >= 0 and np.all(np.abs(data[j - 1, i]) < 1e-5):
            j -= 1
        num_persons[i] = j

    if num_persons.sum() > num_frames and num_persons.sum() < num_frames * 2:
        print('出现杂帧：视频帧中的人数不一直为1或2')

    inpo_data = torch.tensor(np.transpose(data, [0, 2, 3, 1])).contiguous().view(num_person, V * C, num_frames)

    out_results = F.interpolate(inpo_data, size=[tar_len], mode="linear", align_corners=False)

    out_results = np.transpose(out_results.view(num_person, V, C, tar_len).numpy(), [0, 3, 1, 2])

    return out_results


class Feeder(Dataset):
    def __init__(self, train_feeder_args, num_class,
                 random_choose=False, random_shift=False, random_move=False,view_aug=True,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, data_type='train',
                 fill_type=None, frame_num=100, data_random_ob=False, times=1, class_num=60, device=0,
                 dataset='NTU',stream = "body"):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param OBR: Observation Ration,[100,80,60,40,20]
        :param data_type: train or test
        :param frame_num: data frame(100)
        :param times: data repeat times
        :param dataset: Dataset used for training
        """
        self.conn = connect_joint = np.array([2,2,21,3, 21,5,6,7, 21,9,10,11 ,1,13,14,15, 1,17,18,19, 2,23,8,25,12]) - 1
        self.conn_FPHA = connect_joint = np.array([1,1,1,1,1,1,2,7,8,3,10,11,4,13,14,5,16,17,6,19,20])-1

        self.debug = debug
        self.train_feeder_args=train_feeder_args
        self.data_path = train_feeder_args['data_path'][0]
        self.label_path = train_feeder_args['label_path']
        self.dataset=dataset
        self.num_class = num_class
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.times = times
        self.class_num = class_num
        self.device = device
        self.load_data()
        if normalization:
            self.get_mean_map()
        self.OBR = 100
        self.data_type = data_type
        self.fill_type = fill_type
        self.frame_num = frame_num
        self.data_random_ob = data_random_ob
        self.stream = stream
        self.view_aug = view_aug


    def res_random_tag(self):
        self.random_tag = torch.rand(len(self.label))

    def set_OBR(self, OBR_value):
        self.OBR = OBR_value
        return OBR_value

    def load_data(self):
        # data: N M T V C
        #  load label
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                #print(pickle.load(f, encoding='latin1'))
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.label=list(map(int,self.label))

        # load data
        data_type = self.data_path[-3:]
        if data_type == 'npy':
            if self.use_mmap:
                self.data = np.load(self.data_path, mmap_mode='r')
            else:
                self.data = np.load(self.data_path)
        elif data_type == 'pkl':
            try:
                with open(self.data_path) as f2:
                    self.data = pickle.load(f2)
            except:
                # for pickle file from python2
                with open(self.data_path, 'rb') as f2:
                    self.data = pickle.load(f2)

        if len(self.data) != len(self.label):
            raise ValueError("Please ensure they have the same size,data_len{}，label_len{}. ".format(len(self.data),
                                                                                                     len(self.label)))

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        else:
            self.repeat_data(self.times)

        self.index_all = [[i for i, x in enumerate(self.label) if x == j] for j in range(self.num_class)]

    def repeat_data(self, times):
        if times==1:
            return
        N = self.data.shape[0]
        new_data = [val for val in self.data for _ in range(times)]
        new_label = [val for val in self.label for _ in range(times)]
        self.data = new_data
        self.label = new_label

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def label2index(self, inter_class):
        return random.choice(self.index_all[inter_class])

    def get_othersample(self, ori_label):
        tar_label = ori_label
        while tar_label == ori_label:
            tar_label = random.randint(0, 59)
        tar_index = self.label2index(tar_label)

        return tar_index

    def get_othersamples(self, ori_labels):
        data_size = len(ori_labels)
        tar_indexes = torch.zeros(data_size, dtype=torch.int64)
        for i in range(data_size):
            tar_indexes[i] = self.get_othersample(ori_labels[i].item())

        return tar_indexes
    def get_ob_frames(self,data,ration):
        M,T,V,C=data.shape
        ob_frames=[]
        for n in range(data.shape[0]):
            num_frames=0
            for f in range(data.shape[1]):
                if data[n][f].sum()!=0:
                    num_frames=num_frames+1
            ob_frames.append(int(num_frames*ration/100))
        max_obframes=max(ob_frames)
        # if ration != max_obframes:
        #     print(ration, max_obframes)
        if max_obframes !=0:
            return max_obframes
        else:
            return 1
    def UniformSampleFrames(self, ori_data, tar_len, ration=100, fill_type='repeat', visualization=False):
        ob_frames = self.get_ob_frames(ori_data,ration)

        ob_data = ori_data[:,:ob_frames]
        if fill_type == 'empty':
            out_results=ob_data
        elif fill_type == 'zero':
            out_results = ori_data.copy()
            out_results[:, ob_frames:] = 0
        elif fill_type == 'linear':
            if ob_frames < tar_len:
                out_results = FillFrame(ob_data,tar_len=tar_len,device=self.device)
            elif ob_frames > tar_len:
                out_results = CropFrame(ob_data,tar_len=tar_len,device=self.device)
            else:
                out_results = ob_data
        elif fill_type == 'repeat':
            # zero_numpy = np.zeros_like(ori_data)
            out_results = ori_data.copy()
            for i in range(100 - ration):
                out_results[:, ration + i, :, :] = ob_data[:, i % ration, :, :]
        else:
            raise ValueError('fill_type value is zero/linear/repeat,but get:{}'.format(self.test_fill_type))

        # if visualization:
        #     Visualization(out_results)

        return out_results

    def data_repeat(self, ori_data, OB_ration=100):
        num_frames = ori_data.shape[1]
        ob_frames = int(num_frames * OB_ration / 100)
        ob_data = ori_data[:, :ob_frames]
        out_results = ori_data.copy()
        for i in range(num_frames - OB_ration):
            out_results[:, OB_ration + i, :, :] = ob_data[:, i % OB_ration, :, :]
        return out_results

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = int(self.label[index])
        data_numpy = np.array(self.data[index])
        data_numpy_F = data_numpy
        if self.data_type == 'train':
            random_int = int(10 + self.random_tag[index] * 90)
            data_numpy = self.UniformSampleFrames(data_numpy, self.frame_num, random_int, self.fill_type)
        elif self.data_type == 'test' and self.OBR != 100:
            # print('test:index:{}\trandom_int:{}.'.format(index, self.OBR))
            data_numpy = self.UniformSampleFrames(data_numpy, self.frame_num, self.OBR, self.fill_type)


        M, T, V, C = data_numpy.shape

        pairs = ntu_pairs



        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in pairs:
            bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

        joint_vel_data = np.zeros_like(data_numpy)

        bone_vel_data = np.zeros_like(bone_data_numpy)


        joint_vel_data[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        joint_vel_data[:, -1] = 0

        bone_vel_data[:, :-1] = bone_data_numpy[:, 1:] - bone_data_numpy[:, :-1]
        bone_vel_data[:, -1] = 0



        final_data = np.concatenate([data_numpy,bone_data_numpy,joint_vel_data,bone_vel_data], axis=3)

        return final_data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        label = self.label[:len(score)]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
