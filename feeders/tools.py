import random

import numpy as np

# ---------- 基础旋转：按你当前的坐标顺序 [x, z, y] ----------
def _rot_axis(rad, axis):
    c, s = np.cos(rad), np.sin(rad)
    I = np.eye(3, dtype=np.float32)
    R = I.copy()
    if axis == 0:  # 绕 x
        R = np.array([[1, 0, 0],
                      [0, c,-s],
                      [0, s, c]], dtype=np.float32)
    elif axis == 1:  # 绕 z(原始) / 你管线的“竖直轴”
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=np.float32)
    elif axis == 2:  # 绕 y(原始)
        R = np.array([[ c,-s, 0],
                      [ s, c, 0],
                      [ 0, 0, 1]], dtype=np.float32)
    return R

def rotate_xyz(X, yaw=0.0, pitch=0.0, roll=0.0):
    """
    X: (M,T,V,3)；按你当前坐标 [x, z, y]
    yaw   -> 绕 axis=1 旋转（竖直轴，原始 Z）
    pitch -> 绕 axis=0 旋转（原始 X）
    roll  -> 绕 axis=2 旋转（原始 Y）
    """
    if X.size == 0:
        return X
    Ry = _rot_axis(np.deg2rad(yaw),   axis=1)
    Rx = _rot_axis(np.deg2rad(pitch), axis=0)
    Rz = _rot_axis(np.deg2rad(roll),  axis=2)
    R = Rz @ Ry @ Rx
    Y = X.reshape(-1, 3) @ R.T
    return Y.reshape(X.shape)

# ---------- 左右镜像 + 关节重映射 ----------
def leftright_flip(X, flip_pairs, axis=0):
    """
    X: (M,T,V,3)
    flip_pairs: [(l_idx, r_idx), ...]  —— 0-based 关节对
    axis: 镜像的轴分量，按你坐标系一般取 x 轴=0
    """
    if X.size == 0:
        return X
    Y = X.copy()
    Y[..., axis] = -Y[..., axis]
    for l, r in flip_pairs:
        Y[..., [l, r], :] = Y[..., [r, l], :]
    return Y

# ---------- x-z 平面缩放/平移抖动 ----------
def scale_translate_xz(X, sx=1.0, sz=1.0, tx=0.0, tz=0.0, ref_scale=None):
    """
    按你坐标 [x, z, y]：x 索引0，z 索引1
    tx,tz 是“以参考尺度”为单位的平移（默认参考：肩宽）
    """
    if X.size == 0:
        return X
    Y = X.copy()
    Y[..., 0] *= sx   # x
    Y[..., 1] *= sz   # z(竖直)
    if ref_scale is None:
        m0 = 0
        tmid = min(Y.shape[1]//2, Y.shape[1]-1)
        try:
            rsh, lsh = Y[m0, tmid, 8], Y[m0, tmid, 4]  # 0-based: 右肩=8, 左肩=4
            ref_scale = np.linalg.norm(rsh - lsh) + 1e-6
        except Exception:
            ref_scale = np.maximum(1e-3, np.std(Y.reshape(-1, 3), axis=0).mean())
    Y[..., 0] += tx * ref_scale
    Y[..., 1] += tz * ref_scale
    return Y

# ---------- 单调时间重采样（保持长度 T） ----------
def time_resample_keep_len(X, rate=1.0):
    if X.size == 0:
        return X
    M, T, V, C = X.shape
    src_t = np.arange(T, dtype=np.float32)
    dst_t = np.linspace(0, (T-1)/rate, num=T, dtype=np.float32)
    dst_t = np.clip(dst_t, 0.0, T-1.0)

    Y = np.zeros_like(X)
    for m in range(M):
        for v in range(V):
            # 对 3 个坐标分量同时插值：向量化
            Y[m, :, v, :] = np.vstack([
                np.interp(dst_t, src_t, X[m, :, v, c]) for c in range(C)
            ]).T
    return Y

# ---------- 旋转后做一次序列尺度归一（肩宽或身高） ----------
def per_sequence_scale_norm(X, method='shoulder', eps=1e-6):
    if X.size == 0:
        return X
    Y = X.copy()
    if method == 'shoulder':
        m0, tmid = 0, min(Y.shape[1]//2, Y.shape[1]-1)
        try:
            rsh, lsh = Y[m0, tmid, 8], Y[m0, tmid, 4]
            s = np.linalg.norm(rsh - lsh)
        except Exception:
            s = np.linalg.norm(np.nanmax(Y, axis=(0,1)) - np.nanmin(Y, axis=(0,1)))
    else:
        # “身高”粗估：竖直轴（索引1）的范围
        s = float(np.nanmax(Y[..., 1]) - np.nanmin(Y[..., 1]))
    s = max(s, eps)
    Y /= s
    return Y

# ---------- 带偏置的 yaw 采样 ----------
def sample_yaw(mix=[(+35,10), (-35,10), (0,10)], probs=[0.4, 0.4, 0.2], rng=None):
    if rng is None:
        rng = np.random
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / probs.sum()
    idx = rng.choice(len(mix), p=probs)
    mu, sig = mix[idx]
    return float(rng.normal(mu, sig))

# ---------- NTU-25 左右关节对（0-based；请按你的编号核对） ----------
ntu_lr_pairs = [
    (4, 8),   # 左肩, 右肩
    (5, 9),   # 左肘, 右肘
    (6,10),   # 左腕, 右腕
    (11,15),  # 左髋, 右髋
    (12,16),  # 左膝, 右膝
    (13,17),  # 左踝, 右踝
    (7, 3),   # 左手, 右手（若有）
    (14,18),  # 左脚, 右脚（若有）
]

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
