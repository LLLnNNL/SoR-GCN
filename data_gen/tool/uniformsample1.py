import numpy as np
import torch
from torch.nn import functional as F

#from .visualization import Visualization
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

def CropFrame(data,tar_len=100,test_mode=False, num_clips=1,p_interval=(1, 1),float_ok=False,seed=255):
    num_person, num_frames, V, C = data.shape
    if test_mode:
        inds = get_test_clips(num_frames, tar_len, num_clips, p_interval, float_ok, seed)
    else:
        inds = get_train_clips(num_frames, tar_len, num_clips, p_interval, float_ok, seed)

    inds = np.mod(inds, num_frames)

    kp = data.copy()
    #记录每一帧的人数
    if num_person > 1:
        num_persons = [num_person] * num_frames
        for i in range(num_frames):
            j = num_person
            while j >= 0 and np.all(np.abs(kp[j - 1, i]) < 1e-5):
                j -= 1
            num_persons[i] = j

        #找到帧与帧之间人数有变化的位置
        transitional = [False] * num_frames
        for i in range(1, num_frames - 1):
            if num_persons[i] != num_persons[i - 1]:
                transitional[i] = transitional[i - 1] = True
            if num_persons[i] != num_persons[i + 1]:
                transitional[i] = transitional[i + 1] = True
    else:
        transitional = [False] * num_frames

    inds_int = inds.astype(int)
    coeff = np.array([transitional[i] for i in inds_int])
    inds = (coeff * inds_int + (1 - coeff) * inds)

    out_results = np.array([data[:, i, :, :] for i in inds]).transpose([1, 0, 2, 3])
    return out_results[:, :tar_len, :, :]

def FillFrame(data,tar_len=100):
    num_person, num_frames, V, C = data.shape
    # 记录每一帧的人数
    if num_person>1:
        num_persons = np.zeros(num_frames,dtype=int)+2
        for i in range(num_frames):
            j = num_person
            while j >= 0 and np.all(np.abs(data[j - 1, i]) < 1e-5):
                j -= 1
            num_persons[i] = j

        if np.all(num_persons)>num_frames and np.all(num_persons)<num_frames*2 :
            print('出现杂帧：视频帧中的人数不一直为1或2')

    inpo_data=torch.tensor(np.transpose(data, [0, 2, 3, 1])).contiguous().view(num_person,V*C,num_frames)


    out_results = F.interpolate(inpo_data, size=[tar_len], mode="linear", align_corners=False)

    out_results=np.transpose(out_results.view(num_person,V,C,tar_len).numpy(),[0, 3, 1, 2])

    return out_results


def UniformSampleFrames(ori_data, tar_len, ration=1, visualization=False):
    num_frames = ori_data.shape[1]
    ob_frames = int(num_frames * ration)
    ob_data = ori_data[:, :ob_frames]
    if len(ob_data.shape)!=4:
        print('?')
    if ob_frames<tar_len:
        out_results = FillFrame(ob_data,tar_len=tar_len)
    elif ob_frames>tar_len:
        out_results = CropFrame(ob_data,tar_len=tar_len)
    else:
        out_results = ob_data
    # if visualization:
    #     Visualization(out_results)
    return out_results
