import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, total=1000):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.count > total:
            self.sum = 0
            self.count = 0

def write_to_ply(name, points):
    if points.shape[0]>500000:
        return
    # points = points[points[:, 0]!=0, :]
    # print(points.shape)
    with open(name, 'w') as f:
        f.write("""ply
format ascii 1.0
comment zipper output
comment modified by flipply
element vertex {:d}
property float32 x
property float32 y
property float32 z
end_header
""".format(points.shape[0]))
        np.savetxt(f, points, fmt='%.5f')

def write_to_ply_rgb(name, points, clss):
    colorMap = np.array([[ 22,191,206],[214, 38, 40],[ 43,160, 43],[158,216,229],[114,158,206],[204,204, 91],[255,186,119],[147,102,188],[ 30,119,181],[188,188, 33],[255,127, 12],[196,175,214],[153,153,153]])
    colorMap = colorMap.astype(np.int32)
    #clss = clss - 1
    clss = clss.astype(np.int32)

    colors = colorMap[clss]
    points = np.concatenate((points, colors), axis=1)
    #print(points.shape)
    if points.shape[0]>500000:
        return
    # points = points[points[:, 0]!=0, :]
    # print(points.shape)
    with open(name, 'w') as f:
        f.write("""ply
format ascii 1.0
comment zipper output
comment modified by flipply
element vertex {:d}
property float32 x
property float32 y
property float32 z
property uchar red
property uchar green
property uchar blue
end_header
""".format(points.shape[0]))
        for i in range(points.shape[0]):
            f.write('{:.5f} {:.5f} {:.5f} {:d} {:d} {:d}\n'.format(points[i][0], points[i][1], points[i][2], int(points[i][3]), int(points[i][4]), int(points[i][5])))
        #np.savetxt(f, points, fmt='%.5f')

def write_to_ply_color(name, points, colors):
    #colorMap = np.array([[ 22,191,206],[214, 38, 40],[ 43,160, 43],[158,216,229],[114,158,206],[204,204, 91],[255,186,119],[147,102,188],[ 30,119,181],[188,188, 33],[255,127, 12],[196,175,214],[153,153,153]])
    #colorMap = colorMap.astype(np.int32)
    #clss = clss - 1
    #clss = clss.astype(np.int32)

    #colors = colorMap[clss]
    #colors = 
    #print(points.shape, colors.shape)
    points = np.concatenate((points, colors), axis=1)
    #print(points.shape)
    if points.shape[0]>500000:
        return
    # points = points[points[:, 0]!=0, :]
    # print(points.shape)
    with open(name, 'w') as f:
        f.write("""ply
format ascii 1.0
comment zipper output
comment modified by flipply
element vertex {:d}
property float32 x
property float32 y
property float32 z
property uchar red
property uchar green
property uchar blue
end_header
""".format(points.shape[0]))
        for i in range(points.shape[0]):
            f.write('{:.5f} {:.5f} {:.5f} {:d} {:d} {:d}\n'.format(points[i][0], points[i][1], points[i][2], int(points[i][3]), int(points[i][4]), int(points[i][5])))
        #np.savetxt(f, points, fmt='%.5f')


def write_to_pcd(name, points):
    if points.shape[0]>500000:
        return
    with open(name, 'w') as f:
        num = points.shape[0]
        f.write("""
# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {:d}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {:d}
DATA ascii          
""".format(num, num))
        np.savetxt(f, points, fmt='%.5f')

# def write_to_pcd_with_intensity(name, points):
#     if points.shape[0]>500000:
#         return
#     with open(name, 'w') as f:
#         num = points.shape[0]
#         str = """# .PCD v.7 - Point Cloud Data file format
# VERSION .7
# FIELDS x y z intensity
# SIZE 4 4 4 4
# TYPE F F F F
# COUNT 1 1 1 1
# WIDTH {:d}
# HEIGHT 1
# VIEWPOINT 0 0 0 1 0 0 0
# POINTS {:d}
# DATA ascii          
# """
#         str = str.format(num, num)
#         f.write(str)
#         np.savetxt(f, points, fmt='%.5f')

def write_to_pcd_with_intensity(name, points, intensities):
    if points.shape[0]>500000:
        return
    #print(points.shape, intensities[:, np.newaxis].shape)
    points = np.concatenate((points, intensities[:, np.newaxis]), axis=1)
    with open(name, 'w') as f:
        num = points.shape[0]
        str = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {:d}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {:d}
DATA ascii          
"""
        str = str.format(num, num)
        f.write(str)
        np.savetxt(f, points, fmt='%.5f')

def adjust_learning_rate(optimizer, global_counter, batch_size=8, base_lr=0.01, is_debug=False):
    size_step = 100000 / 2
    if is_debug:
        size_step = 1000 * batch_size
    lr = max(base_lr * (0.5 ** (global_counter*batch_size // size_step)), 1e-4)
    #print(global_counter*batch_size, size_step)
    #print(global_counter*batch_size // size_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def point2voxel(point, labels=1, voxel_size=[24, 54, 24]):

    voxel = np.zeros((voxel_size[0], voxel_size[1], voxel_size[2]))
    #point += [12.0, 27.0, 12.0]
    point += np.array(voxel_size, dtype=np.float32) / 2.0
    point += 0.5
    point[point[:,0]<=0, 0] = 0.0
    point[point[:,1]<=0, 1] = 0.0
    point[point[:,2]<=0, 2] = 0.0
    point[point[:,0]>=voxel_size[0], 0] = voxel_size[0] - 1.0
    point[point[:,1]>=voxel_size[1], 1] = voxel_size[1] - 1.0
    point[point[:,2]>=voxel_size[2], 2] = voxel_size[2] - 1.0
    #print('added')
    point = np.floor(point)
    point = point.astype(np.int32)
    voxel[point[:, 0], point[:, 1], point[:, 2]] = labels
    return (voxel)

def points2voxel(points, voxel_size=[24, 54, 24]):
    voxels = np.zeros((points.shape[0], voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.int32)
    for i in range(points.shape[0]):
        voxels[i] = point2voxel(points[i], voxel_size=voxel_size)
    return voxels

def voxel2pc(voxel, voxel_size=[24, 54, 24], num_point=2048*3):
    #result = np.where(np.abs(voxel - 1) < 0.0001)
    result = np.where(voxel==1)
    result_list = list(zip(result[0], result[1], result[2]))
    result = np.array(result_list, dtype=np.float32)
    #result -= [12.0, 27.0, 12.0]
    result -= np.array(voxel_size, dtype=np.float32) / 2.0
    #print(result.shape[0])
    #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))
    choice = np.zeros((num_point,), dtype=np.int32)
    if result.shape[0] > num_point:
        choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=num_point)
    elif result.shape[0] <= num_point:
        choice[:result.shape[0]] = np.arange(result.shape[0])
        choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=num_point - result.shape[0])
    choice.sort()
    result = result[choice, :]

    return result

def voxels2pc(voxels, voxel_size=[24, 54, 24], num_point=2048*3):
    points_out = np.zeros((voxels.shape[0], num_point, 3))
    for i in range(voxels.shape[0]):
        points_out[i] = voxel2pc(voxels[i], voxel_size=voxel_size, num_point=num_point)
    return points_out