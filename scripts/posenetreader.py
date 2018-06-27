from enum import Enum
import h5py
#from data_generators.datausage import DataUsage
from utils.helperclasses import Singleton
from threading import Lock
import numpy as np
import math

class PosenetType(str, Enum):
    TENSORS = 'tensors'
    LABELS = 'labels'

# class PosenetReader(object):
#     def __init__(self, *, data_type, dataset_path_train, dataset_path_test):
#         # reads tensors or labels from the dataset
#         self.data_type = PosenetType(data_type)
#         self.data_holder = PosenetDataHolder(dataset_path_train=dataset_path_train, dataset_path_test=dataset_path_test)
#         self.func_dict = dict()  # a dictionary of functions
#         if self.data_type == PosenetType.TENSORS:   # pass the function handles
#             self.func_dict[DataUsage.TRAINING] = self.data_holder.get_train_tensor
#             self.func_dict[DataUsage.TEST] = self.data_holder.get_test_tensor
#             self.func_dict[DataUsage.VALIDATION] = self.data_holder.get_test_tensor
#         elif self.data_type == PosenetType.LABELS:
#             self.func_dict[DataUsage.TRAINING] = self.data_holder.get_train_label
#             self.func_dict[DataUsage.TEST] = self.data_holder.get_test_label
#             self.func_dict[DataUsage.VALIDATION] = self.data_holder.get_test_label
#
#
#     def __call__(self, datainfo):  # datainfo : keyworded variable length of arguments
#         return self.get_data(**datainfo)  # pass source to source, pass sample_id to sample_id
#
#     def get_data(self, source, sample_id):
#         # source:  train/test/validation
#         # sample_id
#         datausage_mode = DataUsage(source)  # DataUsage.TRAINING / DataUsage.TEST / DataUsage.VALIDATION
#         return self.func_dict[datausage_mode](sample_id)
#
# class PosenetDataHolder(object, metaclass=Singleton):
#     def __init__(self, dataset_path_train, dataset_path_test):
#         with h5py.File(dataset_path_train, 'r') as f:
#             self.x_train = f['/dataset'][()]   # (133,1), array
#             self.y_train = f['/label'][()] # (31,1)
#
#         self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[3],
#                                             self.x_train.shape[2], self.x_train.shape[1])
#         with h5py.File(dataset_path_test, 'r') as f:
#             self.x_test = f['/dataset'][()]
#             self.y_test = f['/label'][()]
#         self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[3],
#                                           self.x_test.shape[2], self.x_test.shape[1])
#     def get_train_tensor(self, sample_id):
#         return self.x_train[sample_id, :, :, :]
#     def get_test_tensor(self, sample_id):
#         return self.x_test[sample_id, :, :, :]
#     def get_train_label(self, sample_id):
#         return self.y_train[sample_id, 0]
#     def get_test_label(self, sample_id):
#         return self.y_test[sample_id, 0]


class PosenetReaderH5(object):
    def __init__(self, *, data_type):
        # reads tensors or labels from the dataset
        self.data_type = PosenetType(data_type)
        self.filename = None
        self.lock = Lock()
        #self.data_holder = PosenetDataHolder(dataset_path_train=dataset_path_train, dataset_path_test=dataset_path_test)
        #self.func_dict = dict()  # a dictionary of functions
        #if self.data_type == PosenetType.TENSORS:   # pass the function handles
        #    self.func_dict[DataUsage.TRAINING] = self.data_holder.get_train_tensor
        #    self.func_dict[DataUsage.TEST] = self.data_holder.get_test_tensor
        #    self.func_dict[DataUsage.VALIDATION] = self.data_holder.get_test_tensor
        #elif self.data_type == PosenetType.LABELS:
        #    self.func_dict[DataUsage.TRAINING] = self.data_holder.get_train_label
        #    self.func_dict[DataUsage.TEST] = self.data_holder.get_test_label
        #    self.func_dict[DataUsage.VALIDATION] = self.data_holder.get_test_label

    def __call__(self, data):
        """

        Parameters
        ----------
        data : dict
            Dictionary containing the filename ("file") of the memorymap, "metadata" describing
            image shape and data type, and a "sample_id" that specifies the desired image index

        Returns
        -------
        An image as numpy array

        """
        with self.lock:
            if self.filename != data["pose_tensor_path"]:
                self.filename = data["pose_tensor_path"]
                with h5py.File(self.filename, 'r') as f:
                    if self.data_type == PosenetType.TENSORS:   # pass the function handles
                        self.x_train = f['/dataset'][()]   # (133,1), array
                        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[3],
                                                            self.x_train.shape[2], self.x_train.shape[1])
                    elif self.data_type == PosenetType.LABELS:
                        self.y_train = f['/label'][()] # (31,1)
        if self.data_type == PosenetType.TENSORS:   # pass the function handles
            return self.x_train[data['sample_id'], :, :, :]
        elif self.data_type == PosenetType.LABELS:
            return self.y_train[data['sample_id'], 0]

def pose_label_minus(x, k):
    x -= k
    return x


class H5Reader(object):
    """
        Class for reading data from a h5 file
    """

    def __init__(self, tensor_min_length):
        self.filename = ""
        self.filedata = None
        self.lock = Lock()
        self.c1_mean = 0
        self.c2_mean = 0
        self.c3_mean = 0
        self.c1_cout = 0
        self.c2_cout = 0
        self.c3_cout = 0
        self.c1_std = 0
        self.c2_std = 0
        self.c3_std = 0
        self.tensor_min_length = tensor_min_length
    def __del__(self):
        """
        Destructor: Close h5 file if open
        """
        # c1_mean = self.c1_mean / self.c1_cout
        # c2_mean = self.c2_mean / self.c2_cout
        # c3_mean = self.c3_mean / self.c3_cout
        # c1_std = self.c1_std / self.c1_cout
        # c2_std = self.c2_std / self.c2_cout
        # c3_std = self.c3_std / self.c3_cout
        try:
            del self.h5file  # close file
        except:
            pass

    def __call__(self, data):
        """

        Parameters
        ----------
        data : dict
            Dictionary containing the filename ("file") of the h5, and a "sample_id" that specifies the desired image index

        Returns
        -------
        pose data of the current frame as numpy array

        """
        with self.lock:
            if self.filename != data["file"]:
                self.filename = data["file"]
                h5filename = self.filename + ".h5"
                try:
                    del self.h5file  # close file
                except:
                    pass
                self.h5file = h5py.File(h5filename, 'r')
                # with h5py.File(h5filename, 'r') as f:
                self.pose_data = self.h5file['/video_pose']
        #            self.pose_data = self.pose_data.reshape(self.pose_data.shape[1], self.pose_data.shape[0])
        #         with h5py.File(h5filename, 'r') as self.h5file:
        #             if self.data_type == PosenetType.TENSORS:  # pass the function handles
        #                 self.x_train = f['/dataset'][()]  # (133,1), array
        #                 self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[3],
        #                                                      self.x_train.shape[2], self.x_train.shape[1])
        #             elif self.data_type == PosenetType.LABELS:
        #                 self.y_train = f['/label'][()]  # (31,1)
        frameIdx = data["sample_id"]
        # tensor normalization parameters
        mean = np.array([66.7, 0, 0])
        std = np.array([7.9, 4.5, 5.2])

        tensor_length = max(self.tensor_min_length, len(frameIdx))
        pose_tensor = np.zeros((tensor_length, self.pose_data.shape[0], 3))
        # default value of the tensor is mean value
        position = len(pose_tensor.shape) - 1
        for i in range(pose_tensor.shape[position]):
            pose_tensor[:, :, i] = mean[i]

        # pose_tensor_aug = np.zeros((len(frameIdx), self.pose_data.shape[0], 3, numAngles))
        #dim1 = pose_tensor.shape(0)
        localIdx = 0
        vel_norm = True

        num_nodes = self.pose_data.shape[0]
        for dim1_id in frameIdx:
        #for dim1_id in range(1, dim1):
            pose_tensor[localIdx, :, 0] = self.pose_data[:, dim1_id]
            # aug_idx = 0
            # pose_xyz = np.array([pose_tensor[localIdx, 0:num_nodes:3, 0], pose_tensor[localIdx, 1:num_nodes:3, 0], pose_tensor[localIdx, 2:num_nodes:3, 0] ])#, pose_tensor[localIdx,1:len(frameIdx):3,0], pose_tensor[localIdx,2#:len(frameIdx):3,0]]
            # for augdegree in degree_range:
            #     theta_aug = augdegree / 180 * math.pi
            #     R_xz_aug = np.array([[math.cos(theta_aug), - math.sin(theta_aug)], [math.sin(theta_aug), math.cos(theta_aug)]])
            #     pose_xyz_rotated = pose_xyz
            #     pose_xyz_rotated[[0, 2], :] = np.dot(R_xz_aug, pose_xyz[[0, 2], :])
            #     pose_tensor_aug[localIdx, :, 0, aug_idx] = np.reshape(pose_xyz_rotated, (num_nodes), order='F')
            #     aug_idx += 1
            if localIdx > 0:
                frame_stride = 1
                pre_idx = frameIdx[localIdx-1]
                if vel_norm:
                    frame_stride = dim1_id - pre_idx
                pose_tensor[localIdx, :, 1] = (self.pose_data[:, dim1_id] - self.pose_data[:, pre_idx]) / frame_stride
            # if localIdx > 0:
            #    pose_tensor[localIdx, :, 2] = (pose_tensor[localIdx, :, 2] - pose_tensor[localIdx - 1, :, 2])
            if localIdx > 1:
                pre_before_last_idx = frameIdx[localIdx-2]
                pose_tensor[localIdx, :, 2] = (-2*self.pose_data[:, pre_idx] + self.pose_data[:, pre_before_last_idx] + self.pose_data[:, dim1_id] ) / (frame_stride*frame_stride)
            localIdx = localIdx + 1
            # pose_tensor_xyz = np.array([pose_tensor[localIdx, 0:num_nodes:3, :], pose_tensor[localIdx, 1:num_nodes:3, :],
            #                             pose_tensor[localIdx, 2:num_nodes:3, :]])
            # for augdegree in degree_range:
            #     theta_aug = augdegree / 180 * math.pi
            #     R_xz_aug = np.array(
            #         [[math.cos(theta_aug), - math.sin(theta_aug)], [math.sin(theta_aug), math.cos(theta_aug)]])
            #     pose_tensor_xyz_rotated = np.array(pose_tensor_xyz)
            #     pose_tensor_xyz_rotated[[0, 2], :, 0] = np.dot(R_xz_aug, pose_tensor_xyz[[0, 2], :, 0])
            #     pose_tensor_xyz_rotated[[0, 2], :, 1] = np.dot(R_xz_aug, pose_tensor_xyz[[0, 2], :, 1])
            #     pose_tensor_xyz_rotated[[0, 2], :, 2] = np.dot(R_xz_aug, pose_tensor_xyz[[0, 2], :, 2])
            #     pose_tensor_aug[localIdx, :, :, aug_idx] = np.reshape(pose_tensor_xyz_rotated, (num_nodes, 3), order='F')
            #     aug_idx += 1

            # if vel_norm:
            #     pose_tensor[localIdx, :, 1] = (pose_tensor[dim1_id, :, 0] - pose_tensor[dim1_id-1, :, 0] )/step
            # else:
            #     pose_tensor[localIdx, :, 1] = (pose_tensor[dim1_id, :, 0] - pose_tensor[dim1_id - 1, :, 0])
            # if acc_norm:
            #     pose_tensor[localIdx, :, 2] = (pose_tensor[dim1_id, :, 1] - pose_tensor[dim1_id - 1, :, 1]) / step
            # else:
            #     pose_tensor[localIdx, :, 2] = (pose_tensor[dim1_id, :, 1] - pose_tensor[dim1_id - 1, :, 1])

        #frame_data = np.array(temp_data)

        position = len(pose_tensor.shape) - 1
        for i in range(pose_tensor.shape[position]):
            pose_tensor[:, :, i] = (pose_tensor[:, :, i] - mean[i]) / std[i]

        # mean_array = np.mean(pose_tensor, axis=0)
        # std_array = np.std(pose_tensor, axis=0)
        # self.c1_mean += np.mean(mean_array[:, 0])
        # self.c2_mean += np.mean(mean_array[:, 1])
        # self.c3_mean += np.mean(mean_array[:, 2])
        # self.c1_std += np.mean(std_array[:, 0])
        # self.c2_std += np.mean(std_array[:, 1])
        # self.c3_std += np.mean(std_array[:, 2])
        # self.c1_cout += 1
        # self.c2_cout += 1
        # self.c3_cout += 1
#
        # self.c1_mean_ = self.c1_mean / self.c1_cout
        # self.c2_mean_ = self.c2_mean / self.c2_cout
        # self.c3_mean_ = self.c3_mean / self.c3_cout
        # self.c1_std_ = self.c1_std / self.c1_cout
        # self.c2_std_ = self.c2_std / self.c2_cout
        # self.c3_std_ = self.c3_std / self.c3_cout
        return pose_tensor
