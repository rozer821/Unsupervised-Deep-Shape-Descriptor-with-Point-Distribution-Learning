from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pickle

class ModelNet(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False):
        print('Loading')

        self.is_train = is_train
        if is_train:
            data = np.load('/home/meng/Code/FolderNet/data/modelNet40_train_file_16nn_GM_cov.npy', encoding='latin1')
        else:
            data = np.load('/home/meng/Code/FolderNet/data/modelNet40_test_file_16nn_GM_cov.npy', encoding='latin1')
        data = data.item()
        label = data['label']
        data = data['data']
        data = data[label==2]
        if is_partial:
            print('partial')
            data[data[:, :, 2]<0] = [0, 0, 0]
            
        self.data = data

        if is_debug:
            self.data = self.data[:12]
    def __getitem__(self, index):
        points = self.data[index]
        return points, index

    def __len__(self):
        return len(self.data)

class Scape(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False):
        print('Loading')

        self.is_train = is_train
        data = np.load('/home/meng/Code/data/scape/data_points.npy')
        if is_train:
            data = data[:50]
        else:
            data = data[50:]
        if is_partial:
            print('partial')
            data[data[:, :, 2]<0] = [0, 0, 0]
            
        self.data = data

        if is_debug:
            self.data = self.data[:12]
    def __getitem__(self, index):
        points = self.data[index]
        return points, index

    def __len__(self):
        return len(self.data)

class ScapeInterpolation(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_normal_noise=False):
        print('Loading')

        self.is_train = is_train
        self.is_normal_noise = is_normal_noise
        #data = np.load('/home/meng/Code/data/scape/data_points_interpolation.npy')
        if is_partial:
            data = np.load('data/tracking/scape_2048_partial.npy')
            #data = np.load('/home/meng/Code/data/scape_select4/select/scape_2048_hole.npy')
        else:
            data = np.load('data/tracking/scape.npy')
        #data = np.load('/home/meng/Code/data/scape_select/scape_8000.npy')
        #data = np.load('/home/meng/Code/data/scape_select/scape_2048_less.npy')
        
        if is_train:
            #data = np.load('/home/meng/Code/data/scape/data_points_interpolation_train.npy')
            #data = np.concatenate((data[0*24:8*24], data[10*24:]), axis=0)
            data = np.concatenate((data[0*24:8*24], data[12*24:]), axis=0)
            print("1111111",len(data))
            import sys
            # sys.exit(0)
            #data = data[0*24:8*24]
        else:
            #data = np.load('/home/meng/Code/data/scape/data_points_interpolation_test.npy')
            data = data[8*24:12*24]
        data[:, :, 0] -= 1.0
        self.data = data

        if is_debug:
            #self.data = self.data[:3]
            self.data = self.data[:12]

    def __getitem__(self, index):
        points = self.data[index]
        if self.is_train:
            points = points.copy()
            choice = np.random.choice(np.arange(2048), size=2048) 
            points = points[choice]

        if self.is_normal_noise:
            points_noise = points.copy()
            points_noise += np.random.randn(points.shape[0], 3) * 0.001
            return points_noise, index, points
        return points, index

    def __len__(self):
        return len(self.data)

class ToscaInterpolation(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False):
        print('Loading')

        self.is_train = is_train
        #data = np.load('/home/meng/Code/data/scape/data_points_interpolation.npy')
        if is_train:
            data = np.load('./data/tosca/data_points_interpolation_cat_train.npy')
            #data = data[:12]
        else:
            data = np.load('./data/tosca/data_points_interpolation_horse_test.npy')
            #data = data[12:]
        data[:, :, 2] -= 50.0
        self.template = data[0]
        if is_partial:
            print('partial')
            data[data[:, :, 0]<0] = [0, 0, 0]
            
        self.data = data

        if is_debug:
            self.data = self.data[:24]
    def __getitem__(self, index):
        points = self.data[index]
        return points, index

    def __len__(self):
        return len(self.data)

class ToscaInterpolationDebug(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_normal_noise=False):
        print('Loading')

        self.is_train = is_train
        self.is_partial = is_partial
        self.is_normal_noise = is_normal_noise
        if is_partial:
            print('partial')
        #data = np.load('/home/meng/Code/data/scape/data_points_interpolation.npy')
        if is_train:
            data = np.load('./data/tosca/data_points_interpolation_cat_train.npy')
            #data = data[:12]
        else:
            data = np.load('./data/tosca/data_points_interpolation_dog_test.npy')
            # data = np.load('./data/tosca/data_points_interpolation_horse_test.npy')
            #data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_dog_test.npy')
            #data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_david_test.npy')
            #data = data[12:]
        data[:, :, 2] -= 50.0
        self.template = data[0]
        
        self.perm = np.random.permutation(data[0].shape[0])   
        self.data = data

        if is_debug:
            self.data = self.data[:24]
    def __getitem__(self, index):
        points = self.data[index]
        if self.is_train:
            points = points.copy()
            choice = np.random.choice(np.arange(2048), size=2048) 
            points = points[choice]
        if self.is_partial:
            #print('partial')
            #print(points.shape)
            points_partial = points.copy()
            sel = points[:, 0]<0
            num = np.sum(sel)
            points_partial[sel] = np.random.randn(num, 3) * 50.0
            #points_partial = points_partial[self.perm, :]
            #points[sel] = np.random.randn(num, 3) * 50.0
            return points_partial, index
        if self.is_normal_noise:
            points_noise = points.copy()
            points_noise += np.random.randn(points.shape[0], 3) * 10.0
            return points_noise, index
        return points, index

    def __len__(self):
        return len(self.data)

class ToscaInterpolationRandom(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False):
        print('Loading')

        self.is_train = is_train
        #data = np.load('/home/meng/Code/data/scape/data_points_interpolation.npy')
        if is_train:
            data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_cat_train.npy')
            #data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_train_random.npy')
            #data = data[:12]
        else:
            #data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_test_random.npy')
            data = np.load('/home/meng/Code/data/tosca/data_points_interpolation_dog_test.npy')
            #data = data[12:]
        # if is_partial:
        #     print('partial')
        #     data[data[:, :, 2]<0] = [0, 0, 0]
        data[:, :, 2] -= 50.0
        self.template = data[0]
        self.data = data

        if is_debug:
            self.data = self.data[:24]
    def __getitem__(self, index):
        points = self.data[index]
        points_shuffle = points.copy()
        np.random.shuffle(points_shuffle)
        return points_shuffle, index

    def __len__(self):
        return len(self.data)

np.random.seed(0)


# class Shapenet(Dataset):
#     def __init__(self, is_train=True, is_debug=False, is_partial=False, is_noise=False):
#         print('Loading')

#         self.is_train = is_train
#         self.is_partial = is_partial
#         self.is_noise = is_noise
#         #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
#         # if self.is_train:
#         #     #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
#         #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
#         # elif self.is_partial:
#         #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_inputs_5_48x64_24x54x24_clean.h5'
#         # else:
#         #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_filled_5_48x64_24x54x24_clean.h5'
#         if self.is_noise:
#             if self.is_train:
#                 #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
#                 file = 'data/sn_noisy/training_prior_filled_5_24x32_24x54x24_noisy.h5'
#             elif self.is_partial:
#                 file = 'data/sn_noisy/test_inputs_5_24x32_24x54x24_noisy.h5'
#             else:
#                 file = 'data/sn_noisy/test_filled_5_24x32_24x54x24_noisy.h5'
#         else:
#             if self.is_train:
#                 #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
#                 file = 'data/sn_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
#             elif self.is_partial:
#                 file = 'data/sn_clean/test_inputs_5_48x64_24x54x24_clean.h5'
#             else:
#                 file = 'data/sn_clean/test_filled_5_48x64_24x54x24_clean.h5'
        
        
        

#         f = h5py.File(file, 'r')
#         #print(f['tensor'].value.shape)
#         self.data = f['tensor'].value
#         print(self.data.shape)

#         # if self.is_train:
#         #     self.data = self.data[:400]
#         # else:
#         #     self.data = self.data[400:]

#         if is_debug:
#             self.data = self.data[:24]
#         self.data = self.data[:, 0]
#         if is_train:
#             temp = np.mean(self.data, axis=0)
#             temp[temp>=0.5] = 1.0
#             #print(temp.shape)
#             #self.template = self.voxel2pc(self.data[0])
#             self.template = self.voxel2pc(temp)

#         self.points = self.get_all_points()
#         print(self.points.shape)


#         #print(self.data.shape)

#     def voxel2pc(self, voxel):
#         # result = np.where(np.abs(voxel-1)<0.0001)
#         result = np.where(voxel==1)
#         result_list = list(zip(result[0], result[1], result[2]))
#         result = np.array(result_list, dtype=np.float32)
#         #print(result.shape)
#         result -= [12.0, 27.0, 12.0]
#         #print(result.shape[0])
#         #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))

#         # choice = np.random.choice(np.arange(result.shape[0]), size=2048*3)
#         # choice.sort()
#         # result = result[choice, :]
#         choice = np.zeros((2048*3,), dtype=np.int32)
#         if result.shape[0] > 2048*3:
#             choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=2048*3)
#         elif result.shape[0] <= 2048*3:
#             choice[:result.shape[0]] = np.arange(result.shape[0])
#             choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=2048*3 - result.shape[0])
#         choice.sort()
#         result = result[choice, :]
#         return result
            
#     def __getitem__(self, index):
#         points = self.data[index]
#         points_new = points.copy()
#         #print(points_new.shape)
#         points_out = self.voxel2pc(points_new)
#         #print(points_new.shape)
#         return points_out, index

#     def get_points(self, idxs):
#         points = self.data[idxs]
#         points_new = points.copy()
#         points_out = np.zeros((len(idxs), 2048*3, 3), dtype=np.float32)
#         for i in range(len(idxs)):
#             points_out[i] = self.voxel2pc(points_new[i])
#         #print(points_new.shape)
#         return points_out

#     def get_all_points(self):
#         voxels = self.data
#         voxels_new = voxels.copy()
#         points_out = np.zeros((self.data.shape[0], 2048*3, 3), dtype=np.float32)
#         for i in range(self.data.shape[0]):
#             points_out[i] = self.voxel2pc(voxels_new[i])
#         #print(points_new.shape)
#         return points_out

#     def __len__(self):
#         return len(self.data)


class ModelnetComp(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_noise=False, voxel_size=[24, 54, 24], num_point=2048*3):
        print('Loading')

        self.is_train = is_train
        self.is_partial = is_partial
        self.is_noise = is_noise
        self.voxel_size = voxel_size
        self.num_point = num_point
        #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # if self.is_train:
        #     #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # elif self.is_partial:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_inputs_5_48x64_24x54x24_clean.h5'
        # else:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_filled_5_48x64_24x54x24_clean.h5'
        #if self.is_noise:
        if self.is_train:
            #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
            file = 'data/modelnet/training_prior_filled_10_64x64_32x32x32_clean.h5'
        elif self.is_partial:
            file = 'data/modelnet/test_inputs_10_64x64_32x32x32_clean.h5'
        else:
            file = 'data/modelnet/test_filled_10_64x64_32x32x32_clean.h5'


        if self.is_train:
            #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
            file = 'data/chair/training_prior_filled_10_64x64_32x32x32_clean.h5'
        elif self.is_partial:
            file = 'data/chair/test_inputs_10_64x64_32x32x32_clean.h5'
        else:
            file = 'data/chair/test_filled_10_64x64_32x32x32_clean.h5'
        # else:
        #     if self.is_train:
        #         #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
        #         file = 'data/sn_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        #     elif self.is_partial:
        #         file = 'data/sn_clean/test_inputs_5_48x64_24x54x24_clean.h5'
        #     else:
        #         file = 'data/sn_clean/test_filled_5_48x64_24x54x24_clean.h5'
        
        
        

        f = h5py.File(file, 'r')
        #print(f['tensor'].value.shape)
        self.data = f['tensor'].value
        print(self.data.shape)

        # if self.is_train:
        #     self.data = self.data[:400]
        # else:
        #     self.data = self.data[400:]

        if is_debug:
            self.data = self.data[:24]
        self.data = self.data[:, 0]
        if is_train:
            temp = np.mean(self.data, axis=0)
            temp[temp>=0.5] = 1.0
            #print(temp.shape)
            #self.template = self.voxel2pc(self.data[0])
            self.template = self.voxel2pc(temp)

        self.points = self.get_all_points()
        print(self.points.shape)


        #print(self.data.shape)

    def voxel2pc(self, voxel):
        # result = np.where(np.abs(voxel-1)<0.0001)
        result = np.where(voxel==1)
        result_list = list(zip(result[0], result[1], result[2]))
        result = np.array(result_list, dtype=np.float32)
        #print(result.shape)
        #result -= [12.0, 27.0, 12.0]
        result -= np.array(self.voxel_size, dtype=np.float32) / 2.0
        #print(result.shape[0])
        #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))

        # choice = np.random.choice(np.arange(result.shape[0]), size=2048*3)
        # choice.sort()
        # result = result[choice, :]
        choice = np.zeros((self.num_point,), dtype=np.int32)
        if result.shape[0] > self.num_point:
            choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=self.num_point)
        elif result.shape[0] <= self.num_point:
            choice[:result.shape[0]] = np.arange(result.shape[0])
            choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=self.num_point - result.shape[0])
        choice.sort()
        result = result[choice, :]
        return result
            
    def __getitem__(self, index):
        points = self.data[index]
        points_new = points.copy()
        #print(points_new.shape)
        points_out = self.voxel2pc(points_new)
        # if self.is_train:
        #     points_out = points_out.copy()
        #     choice = np.random.choice(np.arange(self.num_point), size=self.num_point) 
        #     points_out = points_out[choice]
        #print(points_new.shape)
        return points_out, index

    def get_points(self, idxs):
        points = self.data[idxs]
        points_new = points.copy()
        points_out = np.zeros((len(idxs), self.num_point, 3), dtype=np.float32)
        for i in range(len(idxs)):
            points_out[i] = self.voxel2pc(points_new[i])
        #print(points_new.shape)
        return points_out

    def get_all_points(self):
        voxels = self.data
        voxels_new = voxels.copy()
        points_out = np.zeros((self.data.shape[0], self.num_point, 3), dtype=np.float32)
        for i in range(self.data.shape[0]):
            points_out[i] = self.voxel2pc(voxels_new[i])
        #print(points_new.shape)
        return points_out

    def __len__(self):
        return len(self.data)


noise_add = np.array(np.random.rand(10000, 6144, 3), dtype=np.float32) * 5.0
indices = np.random.choice(np.arange(6144), replace=False, size=5800)
noise_add[:, indices, :] = 0.0


class Shapenet(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_noise=False, voxel_size=[24, 54, 24], is_more_noise=False, num_point=2048*3):
        print('Loading')

        self.is_train = is_train
        self.is_partial = is_partial
        self.is_noise = is_noise
        self.is_more_noise = is_more_noise
        self.voxel_size = voxel_size
        self.num_point = num_point
        #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # if self.is_train:
        #     #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # elif self.is_partial:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_inputs_5_48x64_24x54x24_clean.h5'
        # else:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_filled_5_48x64_24x54x24_clean.h5'
        if self.is_noise:
            if self.is_train:
                #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
                file = 'data/sn_noisy/training_prior_filled_5_24x32_24x54x24_noisy.h5'
            elif self.is_partial:
                file = 'data/sn_noisy/test_inputs_5_24x32_24x54x24_noisy.h5'
            else:
                file = 'data/sn_noisy/test_filled_5_24x32_24x54x24_noisy.h5'
        else:
            if self.is_train:
                #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
                file = 'data/sn_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
            elif self.is_partial:
                file = 'data/sn_clean/test_inputs_5_48x64_24x54x24_clean.h5'
            else:
                file = 'data/sn_clean/test_filled_5_48x64_24x54x24_clean.h5'
        print(file)
        
        
        

        f = h5py.File(file, 'r')
        #print(f['tensor'].value.shape)
        self.data = f['tensor'].value
        print(self.data.shape)

        # if self.is_train:
        #     self.data = self.data[:400]
        # else:
        #     self.data = self.data[400:]

        if is_debug:
            self.data = self.data[:24]
        self.data = self.data[:, 0]
        if is_train:
            temp = np.mean(self.data, axis=0)
            temp[temp>=0.5] = 1.0
            #print(temp.shape)
            #self.template = self.voxel2pc(self.data[0])
            self.template = self.voxel2pc(temp)

        self.points = self.get_all_points()
        print(self.points.shape)


        #print(self.data.shape)

    def voxel2pc(self, voxel):
        # result = np.where(np.abs(voxel-1)<0.0001)
        result = np.where(voxel==1)
        result_list = list(zip(result[0], result[1], result[2]))
        result = np.array(result_list, dtype=np.float32)
        #print(result.shape)
        #result -= [12.0, 27.0, 12.0]
        result -= np.array(self.voxel_size, dtype=np.float32) / 2.0
        #print(result.shape[0])
        #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))

        # choice = np.random.choice(np.arange(result.shape[0]), size=2048*3)
        # choice.sort()
        # result = result[choice, :]
        choice = np.zeros((self.num_point,), dtype=np.int32)
        if result.shape[0] > self.num_point:
            choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=self.num_point)
        elif result.shape[0] <= self.num_point:
            choice[:result.shape[0]] = np.arange(result.shape[0])
            choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=self.num_point - result.shape[0])
        choice.sort()
        result = result[choice, :]
        return result
            
    # def __getitem__(self, index):
    #     points = self.data[index]
    #     points_new = points.copy()
    #     #print(points_new.shape)
    #     points_out = self.voxel2pc(points_new)
    #     #print(points_new.shape)
    #     return points_out, index

    def __getitem__(self, index):
        points = self.points[index]
        points_out = points.copy()
        if self.is_more_noise:
            points_out = noise_add[index] + points_out
        return points_out, index

    def get_points(self, idxs):
        points = self.data[idxs]
        points_new = points.copy()
        points_out = np.zeros((len(idxs), self.num_point, 3), dtype=np.float32)
        for i in range(len(idxs)):
            points_out[i] = self.voxel2pc(points_new[i])
        #print(points_new.shape)
        return points_out

    def get_all_points(self):
        voxels = self.data
        voxels_new = voxels.copy()
        points_out = np.zeros((self.data.shape[0], self.num_point, 3), dtype=np.float32)
        for i in range(self.data.shape[0]):
            points_out[i] = self.voxel2pc(voxels_new[i])
        #print(points_new.shape)
        return points_out

    def __len__(self):
        return len(self.data)

class KITTI(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_noise=False, voxel_size=[24, 54, 24]):
        print('Loading')

        self.is_train = is_train
        self.is_partial = is_partial
        self.is_noise = is_noise
        self.voxel_size = voxel_size
        self.num_point = 2048 * 3
        #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # if self.is_train:
        #     #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        # elif self.is_partial:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_inputs_5_48x64_24x54x24_clean.h5'
        # else:
        #     file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/test_filled_5_48x64_24x54x24_clean.h5'
        
        if self.is_train:
                #file = '/home/meng/Code/data/arxiv2018_shape_completion_clean/training_inference_inputs_5_48x64_24x54x24_clean.h5'
            #file = 'data/kitti/input_training_padding_corrected_1_24x54x24_f.h5'
            file = 'data/sn_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
        elif self.is_partial:
            file = 'data/kitti/input_validation_gt_padding_corrected_1_24x54x24_f.h5'
        else:
            file = 'data/kitti/input_validation_gt_padding_corrected_1_24x54x24_f.h5'
        
        

        f = h5py.File(file, 'r')
        #print(f['tensor'].value.shape)
        self.data = f['tensor'].value
        print(self.data.shape)

        # if self.is_train:
        #     self.data = self.data[:400]
        # else:
        #     self.data = self.data[400:]

        if is_debug:
            self.data = self.data[:24]
        self.data = self.data[:, 0]
        if is_train:
            temp = np.mean(self.data, axis=0)
            temp[temp>=0.5] = 1.0
            #print(temp.shape)
            #self.template = self.voxel2pc(self.data[0])
            self.template = self.voxel2pc(temp)

        self.points = self.get_all_points()
        print(self.points.shape)


        #print(self.data.shape)

    def voxel2pc(self, voxel):
        # result = np.where(np.abs(voxel-1)<0.0001)
        #print(voxel)
        #print(np.amax(voxel))
        result = np.where(voxel>0)
        result_list = list(zip(result[0], result[1], result[2]))
        result = np.array(result_list, dtype=np.float32)
        #print(result.shape)
        #result -= [12.0, 27.0, 12.0]
        #print(result.shape)
        result -= np.array(self.voxel_size, dtype=np.float32) / 2.0
        #print(result.shape[0])
        #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))

        # choice = np.random.choice(np.arange(result.shape[0]), size=2048*3)
        # choice.sort()
        # result = result[choice, :]
        choice = np.zeros((self.num_point,), dtype=np.int32)
        if result.shape[0] > self.num_point:
            choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=self.num_point)
        elif result.shape[0] <= self.num_point:
            choice[:result.shape[0]] = np.arange(result.shape[0])
            choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=self.num_point - result.shape[0])
        choice.sort()
        result = result[choice, :]
        return result
            
    def __getitem__(self, index):
        points = self.data[index]
        points_new = points.copy()
        #print(points_new.shape)
        points_out = self.voxel2pc(points_new)
        #print(points_new.shape)
        return points_out, index

    def get_points(self, idxs):
        points = self.data[idxs]
        points_new = points.copy()
        points_out = np.zeros((len(idxs), self.num_point, 3), dtype=np.float32)
        for i in range(len(idxs)):
            points_out[i] = self.voxel2pc(points_new[i])
        #print(points_new.shape)
        return points_out

    def get_all_points(self):
        voxels = self.data
        voxels_new = voxels.copy()
        points_out = np.zeros((self.data.shape[0], self.num_point, 3), dtype=np.float32)
        for i in range(self.data.shape[0]):
            points_out[i] = self.voxel2pc(voxels_new[i])
        #print(points_new.shape)
        return points_out

    def __len__(self):
        return len(self.data)


class FAUST(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_normal_noise=False):
        print('Loading')

        self.is_train = is_train
        data = np.load('/home/meng/mmvc_large/Meng_Wang/main_comp/train_points.npy')
        # if is_train:
        #     #data = np.load('/home/meng/Code/data/scape/data_points_interpolation_train.npy')
        #     data = data[0*24:10*24]
        # else:
        #     #data = np.load('/home/meng/Code/data/scape/data_points_interpolation_test.npy')
        #     data = data[12*24:14*24]
        #data[:, :, 0] -= 1.0
        self.data = data

        if is_debug:
            self.data = self.data[:24]
        self.points_all = self.rearange()

    def rearange(self):
        points_all = np.zeros((3 * (self.data.shape[0] - 2), self.data.shape[1], 3), dtype=np.float32)
        for i in range(self.data.shape[0] - 2):
            points_all[3*i] = self.data[i]
            points_all[3*i + 1] = self.data[i + 1]
            points_all[3*i + 2] = self.data[i + 2]
        return points_all


    def __getitem__(self, index):
        points = self.data[index]
        # if self.is_normal_noise:
        #     points_noise = points.copy()
        #     points_noise += np.random.randn(points.shape[0], 3) * 0.04
        #     return points_noise, index
        return points, index

    def __len__(self):
        return len(self.data)

class KITTI_Velo(Dataset):
    def __init__(self, is_train=True, is_debug=False, is_partial=False, is_noise=False, voxel_size=[24, 54, 24]):
        print('Loading')

        self.is_train = is_train
        self.is_partial = is_partial
        self.is_noise = is_noise
        self.voxel_size = voxel_size
        self.num_point = 2048 * 3
        
        if self.is_train:
            file = 'data/sn_clean/training_prior_filled_5_48x64_24x54x24_clean.h5'
            f = h5py.File(file, 'r')
            self.data = f['tensor'].value
            print(self.data.shape)
            if is_debug:
                self.data = self.data[:24]
            self.data = self.data[:, 0]
            if is_train:
                temp = np.mean(self.data, axis=0)
                temp[temp>=0.5] = 1.0
                self.template = self.voxel2pc(temp)
            self.points = self.get_all_points()
            print(self.points.shape)
            self.idx_all = np.zeros((self.points.shape[0]), dtype=np.float32)
            self.idx_obj_all = np.zeros((self.points.shape[0]), dtype=np.float32)
        else:
            with open('/home/meng/Code/frustum-pointnets/kitti/debug/data_kitti_debug.pickle', 'rb') as handle:
                points_all = pickle.load(handle)
                idx_all = pickle.load(handle)
                idx_obj_all =pickle.load(handle) 
            points_all_np = np.zeros((len(points_all), self.num_point, 3), dtype=np.float32)
            for i in range(len(points_all)):
                point = points_all[i]
                choice = np.random.choice(np.arange(point.shape[0]), size=self.num_point)
                point = point[choice]
                points_all_np[i] = point

            self.points = points_all_np
            self.idx_all = np.array(idx_all, dtype=np.float32)
            self.idx_obj_all = np.array(idx_obj_all, dtype=np.float32)
            print(self.points.shape)



        

        #print(self.data.shape)

    def voxel2pc(self, voxel):
        # result = np.where(np.abs(voxel-1)<0.0001)
        #print(voxel)
        #print(np.amax(voxel))
        result = np.where(voxel>0)
        result_list = list(zip(result[0], result[1], result[2]))
        result = np.array(result_list, dtype=np.float32)
        #print(result.shape)
        #result -= [12.0, 27.0, 12.0]
        #print(result.shape)
        result -= np.array(self.voxel_size, dtype=np.float32) / 2.0
        #print(result.shape[0])
        #print(np.amax(result[:, 0]), np.amax(result[:, 1]), np.amax(result[:, 2]))

        # choice = np.random.choice(np.arange(result.shape[0]), size=2048*3)
        # choice.sort()
        # result = result[choice, :]
        choice = np.zeros((self.num_point,), dtype=np.int32)
        if result.shape[0] > self.num_point:
            choice[:result.shape[0]] = np.random.choice(np.arange(result.shape[0]), size=self.num_point)
        elif result.shape[0] <= self.num_point:
            choice[:result.shape[0]] = np.arange(result.shape[0])
            choice[result.shape[0]:] = np.random.choice(np.arange(result.shape[0]), size=self.num_point - result.shape[0])
        choice.sort()
        result = result[choice, :]
        return result
            
    def __getitem__(self, index):
        point = self.points[index]
        idx = self.idx_all[index]
        idx_obj = self.idx_obj_all[index]
        return point, index, idx, idx_obj

    def get_points(self, idxs):
        points = self.data[idxs]
        points_new = points.copy()
        points_out = np.zeros((len(idxs), self.num_point, 3), dtype=np.float32)
        for i in range(len(idxs)):
            points_out[i] = self.voxel2pc(points_new[i])
        #print(points_new.shape)
        return points_out

    def get_all_points(self):
        voxels = self.data
        voxels_new = voxels.copy()
        points_out = np.zeros((self.data.shape[0], self.num_point, 3), dtype=np.float32)
        for i in range(self.data.shape[0]):
            points_out[i] = self.voxel2pc(voxels_new[i])
        #print(points_new.shape)
        return points_out

    def __len__(self):
        return (self.points.shape[0])


