import numpy as np
import argparse
import os.path as osp
import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

from tool.read_xyz import read_xyz
from tool.preprocess import gen_oridata
from tool.preprocess import pre_normalization
from tool.uniformsample1 import UniformSampleFrames
from tool.visualization import Visualization

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
training_cameras = [2, 3]
observation_ration = [1]
max_body = 2
num_joint = 25
max_frame = 100
toolbar_width = 30


def get_parser():
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='./data/nturgbd_raw/nturgb+d_skeletons')  # 导入数据集源位置
    parser.add_argument(
        '--ignored_sample_path',
        default='./data_gen/samples_with_missing_skeletons.txt')  # 忽略不处理的源文件名称
    parser.add_argument('--out_folder', default='./data/NTU-RGB-D')  # 处理后数据集存储位置
    parser.add_argument('--gen_ori', default=False)  # 处理后数据集存储位置gen_ori
    return parser


def check_samples(filename, benchmark, part):
    # S001C003P002R001A0011:action_class=011;subject_id=002;camera_id=003
    action_class = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    # 根据 benchmark 要求划分训练集与验证集
    if benchmark == 'xview':
        istraining = (camera_id in training_cameras)
    elif benchmark == 'xsub':
        istraining = (subject_id in training_subjects)
    else:
        raise ValueError()
    # 判断此样本是否为训练集（验证集）样本
    if part == 'train':
        issample = istraining
    elif part == 'val':
        issample = not (istraining)
    else:
        raise ValueError()
    return issample, action_class

def show_dataframe(data_path,ignored_sample_path):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []

    sample_name = []  # 样本名称
    sample_label = []  # 样本标签
    sample_frame=[]
    # 导入数据集源文件名称
    item=0
    for filename in tqdm(os.listdir(data_path)):
        if filename in ignored_samples:
            continue
        data = read_xyz(os.path.join(data_path, filename), max_body=max_body, num_joint=num_joint)
        sample_frame.append(data.shape[1])
    print(sample_frame)
    a = np.array(sample_frame)
    np.save('NTU_RGBD_frames.npy', a)  # 保存为.npy格式
    plt.style.use('fivethirtyeight')

    plt.hist(sample_frame)

    plt.title('NTU RGB+D')
    plt.xlabel('frame number')
    plt.ylabel('frequency')

    plt.tight_layout()

    plt.show()

def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xsub', part='val',gen_ori=False):
    # 读取数据集源文件名称
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []

    sample_name = []  # 样本名称
    sample_label = []  # 样本标签

    # 导入数据集源文件名称
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        issample, action_class = check_samples(filename, benchmark, part)
        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
    # 存储 label 信息
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    # 文件暂存对象
    fp = np.zeros((len(sample_label), max_body, max_frame, num_joint, 3), dtype=np.float32)
    fp_ori = np.zeros((len(sample_label), max_body, max_frame, num_joint, 3), dtype=np.float32)
    for ration in observation_ration:
        print('观察比例：{}'.format(ration))
        for i, s in enumerate(tqdm(sample_name)):
            # 存入时空图信息
            data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            #Visualization(data,transpose=False)
            if gen_ori:
                ori_data = gen_oridata(data,max_frame)
                # Visualization(ori_data)
                fp_ori[i] = ori_data
            PN_data = pre_normalization(data, sample_name[i])
            #Visualization(PN_data)
            visualization = True if i == 0 else False
            USF_data = UniformSampleFrames(PN_data, max_frame, ration)
            # Visualization(USF_data)
            fp[i] = USF_data

        print("save:{}_{}_data_joint.npy".format(part,int(ration*100)))
        np.save('{}/{}_{}_data_joint.npy'.format(out_path, part,int(ration*100)), fp)
        if gen_ori:
            print("save:{}_{}_data_joint_ori.npy".format(part,int(ration*100)))
            np.save('{}/{}_{}_data_joint_ori.npy'.format(out_path, part,int(ration*100)), fp_ori)


if __name__ == '__main__':

    parser = get_parser()

    benchmark = ['xview','xsub']
    part = ['train','val']
    arg = parser.parse_args()

    # show_dataframe(arg.data_path,arg.ignored_sample_path)

    for b in benchmark:
        for p in part:
            out_path = osp.join(arg.out_folder, b, p)
            print('{} {} 数据保存于：{}'.format(b, p, osp.abspath(out_path)))
            os.makedirs(out_path, exist_ok=True)
            gendata(arg.data_path, out_path, arg.ignored_sample_path, benchmark=b, part=p,gen_ori=arg.gen_ori)
