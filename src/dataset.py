from src.utils import pdump, pload, bmtv, bmtm, bmv, pltt, plts
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import sys


class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, dt):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')

        self.mode = mode
        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs,
            test_seqs)
        # get and compute value for normalizing inputs
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)
        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # IMU sampling time
        self.dt = dt # (s)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1),
            torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        """Get IMU input and ground-truth ZUPT"""
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        u = mondict['us']
        x = mondict['xs']
        return u, x

    def __len__(self):
        return len(self.sequences)

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors not computed")
            return 0, 0

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError
        
    def get_test(self, i):
        input_dict = self.load_seq(i)
        gt_zupts = input_dict['xs'][:, 0]
        us = input_dict['us']
        gt_zupts[0] = 0
        Nshift = torch.where(gt_zupts == 1)[0][0]
        N = us.shape[0] - Nshift
        ts = torch.linspace(0, (N-1)*self.dt, N)
        return ts, us, Nshift


class KaistDataset(BaseDataset):
    """
        Dataloader for the Kaist Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, dt):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        # threshold for ZUPT ground truth
        sm_velocity_max_threshold = 0.004  # m/s

        f = os.path.join(self.predata_dir, 'urban06.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "sensor_data",
                "xsens_imu.csv")
            path_gt = os.path.join(data_dir, seq, "global_pose.csv")
            # path_odo = os.path.join(data_dir, seq, "encoder.csv")
            return path_imu, path_gt

        time_factor = 1e9  # ns -> s

        def interpolate(x, t, t_int, angle=False):
            """
            Interpolate ground truth with sensors
            """
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if angle:
                    x[:, i] = np.unwrap(x[:, i])
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            return x_int

        sequences = os.listdir(data_dir)
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",")

            # Urban00-05 and campus00 have only quaternion and Euler data
            if not imu.shape[1] > 10:
                cprint("No IMU data for dataset " + sequence, 'yellow')
                continue
            gt = np.genfromtxt(path_gt, delimiter=",")

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            t = imu[:, 0]

            # take ground truth position
            p_gt = gt[:, [4, 8, 12]]
            p_gt = p_gt - p_gt[0]

            # take ground matrix pose
            Rot_gt = torch.Tensor(gt.shape[0], 3, 3)
            for j in range(3):
                Rot_gt[:, j] = torch.Tensor(gt[:, 1 + 4 * j: 1 + 4 * j + 3])
            q_gt = SO3.to_quaternion(Rot_gt)
            # convert to angle orientation
            rpys = SO3.to_rpy(Rot_gt)
            t_gt = gt[:, 0]
            # interpolate ground-truth
            p_gt = interpolate(p_gt, t_gt, t)
            rpys = interpolate(rpys.numpy(), t_gt, t, angle=True)

            # convert from numpy
            ts = (t - t0)/time_factor
            p_gt = torch.Tensor(p_gt)
            rpys = torch.Tensor(rpys).float()
            q_gt = SO3.to_quaternion(SO3.from_rpy(rpys[:, 0], rpys[:, 1], rpys[:, 2]))
            imu = torch.Tensor(imu).float()

            # take IMU gyro and accelerometer and magnetometer
            imu = imu[:, 8:17]

            dt = ts[1:] - ts[:-1]
            # compute speed ground truth (apply smoothing)
            v_gt = torch.zeros(p_gt.shape[0], 3)
            for j in range(3):
                p_gt_smooth = savgol_filter(p_gt[:, j], 11, 1)
                v_j = (p_gt_smooth[1:] - p_gt_smooth[:-1]) / dt
                v_j_smooth = savgol_filter(v_j, 11, 0)
                v_gt[1:, j] = torch.Tensor(v_j_smooth)

            # ground truth specific motion measurement (binary)
            zupts = v_gt.norm(dim=1, keepdim=True) < sm_velocity_max_threshold
            zupts = zupts.float()
            # set ground truth consistent with ZUPT
            v_gt[zupts.squeeze() == 1] = 0

            # save for all training
            mondict = {
                'xs': zupts.float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")
