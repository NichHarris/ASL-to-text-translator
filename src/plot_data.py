import torch
import time
import matplotlib.pyplot as plt

def arr_subset(x_arr, y_arr, z_arr, start_range, end_range):
    return x_arr[start_range:end_range], z_arr[start_range:end_range], y_arr[start_range:end_range]

def arr_connect(x_arr, y_arr, z_arr, pt_1, pt_2):
    x_temp = []
    x_temp.append(x_arr[pt_1])
    x_temp.append(x_arr[pt_2])

    y_temp = []
    y_temp.append(y_arr[pt_1])
    y_temp.append(y_arr[pt_2])

    z_temp = []
    z_temp.append(z_arr[pt_1])
    z_temp.append(z_arr[pt_2])

    return x_temp, z_temp, y_temp

def visualize_hand(xs, ys, zs):
    # 0-4
    ax.plot(*arr_subset(xs, zs, ys, 0, 5), c='black')
    #0, 5-8
    ax.plot(*arr_subset(xs, zs, ys, 5, 9), c='black')
    #5, 9-12
    ax.plot(*arr_subset(xs, zs, ys, 9, 13), c='black')
    #9, 13-16
    ax.plot(*arr_subset(xs, zs, ys, 13, 17), c='black')
    #13, 17-20
    ax.plot(*arr_subset(xs, zs, ys, 17, 21), c='black')

    #0, 5
    ax.plot(*arr_connect(xs, zs, ys, 0, 5), c='black')
    #5, 9
    ax.plot(*arr_connect(xs, zs, ys, 5, 9), c='black')
    #9, 13
    ax.plot(*arr_connect(xs, zs, ys, 9, 13), c='black')
    #13, 17
    ax.plot(*arr_connect(xs, zs, ys, 13, 17), c='black')
    #17, 0
    ax.plot(*arr_connect(xs, zs, ys, 0, 17), c='black')

def visualize_pose(xs, ys, zs):
    # 0-3
    ax.plot(*arr_subset(xs, zs, ys, 0, 4), c='black')
    #0,4-6
    ax.plot(*arr_subset(xs, zs, ys, 4, 7), c='black')
    #9, 10
    ax.plot(*arr_subset(xs, zs, ys, 9, 11), c='black')
    #11, 12
    ax.plot(*arr_subset(xs, zs, ys, 11, 13), c='black')

    #3, 7
    ax.plot(*arr_connect(xs, zs, ys, 3, 7), c='black')
    #6, 8
    ax.plot(*arr_connect(xs, zs, ys, 6, 8), c='black')
    #11, 13
    ax.plot(*arr_connect(xs, zs, ys, 11, 13), c='black')
    #11, 23
    ax.plot(*arr_connect(xs, zs, ys, 11, 23), c='black')
    #12, 24
    ax.plot(*arr_connect(xs, zs, ys, 12, 24), c='black')
    # 12, 14
    ax.plot(*arr_connect(xs, zs, ys, 12, 14), c='black')
    # 14, 16
    ax.plot(*arr_connect(xs, zs, ys, 14, 16), c='black')
    # 16, 18
    ax.plot(*arr_connect(xs, zs, ys, 16, 18), c='black')
    # 18, 20
    ax.plot(*arr_connect(xs, zs, ys, 18, 20), c='black')
    # 16, 20
    ax.plot(*arr_connect(xs, zs, ys, 16, 20), c='black')    
    # 16, 22
    ax.plot(*arr_connect(xs, zs, ys, 16, 22), c='black')      
    # 13, 15
    ax.plot(*arr_connect(xs, zs, ys, 13, 15), c='black')
    # 15, 21
    ax.plot(*arr_connect(xs, zs, ys, 15, 21), c='black')
    # 15, 17
    ax.plot(*arr_connect(xs, zs, ys, 15, 17), c='black')
    # 15, 19
    ax.plot(*arr_connect(xs, zs, ys, 15, 19), c='black')    
    # 19, 17
    ax.plot(*arr_connect(xs, zs, ys, 17, 19), c='black')   

# Main script
frames = torch.load(f'../dataset_only/bye_bye_0.pt')
print("-- Frames --")
fig = plt.figure(figsize = (8,8))
for i, frame in enumerate(frames):
    ax = plt.axes(projection='3d')

    xs = []
    ys = []
    zs = []

    left_hand = frame[:21*3]
    right_hand = frame[21*3:21*3*2]
    pose = frame[21*3*2:]

    xyz = iter(pose)
    for x, y, z, v in zip(xyz, xyz, xyz, xyz):
        xs.append(x)
        ys.append(y)
        zs.append(z)

    
    ax.scatter(xs, zs, ys, c='red', s=50)
    visualize_pose(xs, zs, ys)
    ax.invert_zaxis()
    ax.view_init(elev=10, azim=-70)


    # TODO: Fix visualize hand
    # xyz = iter(right_hand)
    # for x, y, z in zip(xyz, xyz, xyz):
    #     xs.append(x)
    #     ys.append(y)
    #     zs.append(z)  

    #visualize_hand(xs, ys, zs)

    # Visualize sign from coordinates
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()