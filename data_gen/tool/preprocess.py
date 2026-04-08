from .rotation import *
from .visualization import Visualization
def pre_normalization(data, filename, frame_num=100, center=1,zaxis=[0, 1], xaxis=[8, 4],channel=3):
    total_frames = data.shape[1]
    C, T, V, M = data.shape
    data_tran = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to   M, T, V, C

    assert T == total_frames
    if data_tran.sum() == 0:
        print('pad :has no skeleton')
        return

    index0 = [i for i in range(T) if not np.all(np.isclose(data_tran[0, i], 0))]

    assert M in [1, 2]
    if M == 2:
        index1 = [i for i in range(T) if not np.all(np.isclose(data_tran[1, i], 0))]
        #如果单人动作没有保存在下标0,则换过来
        #删除空白帧
        if len(index0) < len(index1):
            skeleton = data_tran[:, np.array(index1)]
            skeleton = skeleton[[1, 0]]
        else:
            skeleton = data_tran[:, np.array(index0)]
    else:
        skeleton = data_tran[:, np.array(index0)]

    T_new = skeleton.shape[1]

    #align_center
    main_body_center = skeleton[0][:, center:center+1, :].copy()
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        try:
            mask = (person.sum(-1) != 0).reshape(T_new, V, 1)
        except:
            print('数据异常')
        skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask

    #align_spine
    if channel==3:
        if zaxis!= -1 :
            joint_bottom = skeleton[0, 0, zaxis[0]]
            joint_top = skeleton[0, 0, zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)
            skeleton = skeleton[:, :, :, [0, 2, 1]]
        if xaxis != -1:
            joint_rshoulder = skeleton[0, 0, xaxis[0]]
            joint_lshoulder = skeleton[0, 0, xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)
    elif channel == 2:
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 1])
        rmatrix = rotation_matrix(axis, angle,channel=2)
        skeleton = np.einsum('abcd,kd->abck', skeleton, rmatrix)
    return skeleton
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

def CropFrame(data,tar_len=100,test_mode=False, num_clips=1,p_interval=(1, 1),float_ok=False,seed=255):
    num_person, num_frames, V, C = data.shape

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

    inds_int = inds.astype(np.int)
    coeff = np.array([transitional[i] for i in inds_int])
    inds = (coeff * inds_int + (1 - coeff) * inds)

    out_results = np.array([data[:, i, :, :] for i in inds]).transpose([1, 0, 2, 3])
    return out_results[:, :tar_len, :, :]

def FillFrame(data,tar_len=100):
    num_person, T, V, C = data.shape
    pad_frames=int(tar_len - T)
    out_results=np.pad(data,((0,0),(0,pad_frames),(0,0),(0,0)))
    return out_results
def gen_oridata(data, frame_num):
    total_frames = data.shape[1]
    C, T, V, M = data.shape
    data_tran = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to   M, T, V, C

    assert T == total_frames
    if data.sum() == 0:
        print('pad :has no skeleton')
        return
    if T<frame_num:
        skeleton = FillFrame(data_tran,tar_len=frame_num)
    elif T>frame_num:
        skeleton = CropFrame(data_tran,tar_len=frame_num)
    else:
        skeleton = data_tran
    # skeleton=skeleton[:, :, :, [0, 2, 1]]
    return skeleton

