import os
import numpy as np
import matplotlib.pyplot as plt


## 2D展示
def Print2D_ntu(num_frame, point, arms, rightHand, leftHand, legs, body):
    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    n = 0  # 从第n帧开始展示
    m = num_frame  # 到第m帧结束，n<m<row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla()  # # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

        # 画出两个body所有关节
        plt.scatter(point[0, i, :, :], point[1, i, :, :], c='red', s=40.0)  # c: 颜色;  s: 大小

        # 连接第一个body的关节，形成骨骼
        plt.plot(point[0, i, arms, 0], point[1, i, arms, 0], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c='green', lw=2.0)  # c: 颜色;  lw: 线条宽度
        plt.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 0], point[1, i, legs, 0], c='green', lw=2.0)
        plt.plot(point[0, i, body, 0], point[1, i, body, 0], c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        plt.plot(point[0, i, arms, 1], point[1, i, arms, 1], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 1], point[1, i, legs, 1], c='green', lw=2.0)
        plt.plot(point[0, i, body, 1], point[1, i, body, 1], c='green', lw=2.0)

        plt.text(xmax, ymax + 0.2, 'frame: {}/{}'.format(i, num_frame - 1))  # 文字说明
        plt.xlim(xmin - 0.5, xmax + 0.5)  # x坐标范围
        plt.ylim(ymin - 0.3, ymax + 0.3)  # y坐标范围
        plt.pause(0.001)  # 停顿延时

    plt.ioff()
    plt.show()

def Print3D_ntu(num_frame, point, arms, rightHand, leftHand, legs, body):
    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    n = 0  # 从第n帧开始展示
    m = num_frame  # 到第m帧结束，n<m<row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla()  # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

        plot3D = plt.subplot(projection='3d')
        plot3D.view_init(120, -90)  # 改变视角

        Expan_Multiple = 1.4  # 坐标扩大倍数，绘图较美观

        # 画出两个body所有关节
        plot3D.scatter(point[0, i, :, :] * Expan_Multiple, point[1, i, :, :] * Expan_Multiple, point[2, i, :, :],
                       c='red', s=40.0)  # c: 颜色;  s: 大小

        # 连接第一个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 0] * Expan_Multiple, point[1, i, arms, 0] * Expan_Multiple, point[2, i, arms, 0],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0] * Expan_Multiple, point[1, i, rightHand, 0] * Expan_Multiple,
                    point[2, i, rightHand, 0], c='green', lw=2.0)  # c: 颜色;  lw: 线条宽度
        plot3D.plot(point[0, i, leftHand, 0] * Expan_Multiple, point[1, i, leftHand, 0] * Expan_Multiple,
                    point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0] * Expan_Multiple, point[1, i, legs, 0] * Expan_Multiple, point[2, i, legs, 0],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0] * Expan_Multiple, point[1, i, body, 0] * Expan_Multiple, point[2, i, body, 0],
                    c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 1] * Expan_Multiple, point[1, i, arms, 1] * Expan_Multiple, point[2, i, arms, 1],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 1] * Expan_Multiple, point[1, i, rightHand, 1] * Expan_Multiple,
                    point[2, i, rightHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 1] * Expan_Multiple, point[1, i, leftHand, 1] * Expan_Multiple,
                    point[2, i, leftHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 1] * Expan_Multiple, point[1, i, legs, 1] * Expan_Multiple, point[2, i, legs, 1],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 1] * Expan_Multiple, point[1, i, body, 1] * Expan_Multiple, point[2, i, body, 1],
                    c='green', lw=2.0)

        plot3D.text(xmax - 0.3, ymax + 1.1, zmax + 0.3, 'frame: {}/{}'.format(i, num_frame - 1))  # 文字说明
        plot3D.set_xlim3d(xmin - 0.5, xmax + 0.5)  # x坐标范围
        plot3D.set_ylim3d(ymin - 0.3, ymax + 0.3)  # y坐标范围
        plot3D.set_zlim3d(zmin - 0.3, zmax + 0.3)  # z坐标范围
        plt.pause(0.001)  # 停顿延时

    plt.ioff()
    plt.show()

def Visualization(data, transpose=True,dataset='ntu'):
    #input:[C,T,V,M]
    num_frame = data.shape[1]  # 帧数
    if transpose == True:
        data = np.transpose(data, [3, 1, 2, 0])
    print(data.shape)  # 坐标数(3) × 帧数 × 关节数(25) × max_body(2)
    if dataset=='NTU':
        # 相邻关节标号
        arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]  # 23 <-> 11 <-> 10 ...
        rightHand = [11, 24]  # 11 <-> 24
        leftHand = [7, 22]  # 7 <-> 22
        legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]  # 19 <-> 18 <-> 17 ...
        body = [3, 2, 20, 1, 0]  # 3 <-> 2 <-> 20 ...
        Print3D_ntu(num_frame, data, arms, rightHand, leftHand, legs, body)  # 3D可视化
    else :
        print("There is no visualization of the {} data set".format(dataset))



