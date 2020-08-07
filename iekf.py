import torch
from src.utils import pdump, pload, bmmt, axat, pltt, plts
import matplotlib.pyplot as plt
import os
import numpy as np
from termcolor import cprint
from src.lie_algebra import SO3


class BaseIEKF:
    """IEKF where variable are in CPU, in double, and NOT in batch.
    Record only position."""

    # gravity vector
    g = torch.Tensor([0, 0, -9.80665]).double()

    # identity matrices
    Id3 = g.new_ones(3).diag()
    IdP = g.new_ones(15).diag()

    # Jacobians
    F = g.new_zeros(15, 15)
    G = g.new_zeros(15, 12)
    G[9:12, 6:9] = Id3
    G[12:15, 9:12] = Id3
    H = g.new_zeros(9, 15)

    def __init__(self, th_max_zupt, th_min_zupt, max_omega_norm,
        max_omega, max_acc_norm, max_acc, zupt_omega_std, zupt_acc_std,
        N_init, N_normalize):

        # initial covariance
        self.P0, self.Q = self.init_cov()

        # each ZUPT outsides this threshold set ZUPT == 1
        self.th_max_zupt = th_max_zupt
        # each ZUPT outsides this threshold set ZUPT == 0
        self.th_min_zupt = th_min_zupt

        # standard deviation on ZUPT IMU measurements
        self.zupt_omega_std = zupt_omega_std
        self.zupt_acc_std = zupt_acc_std

        self.zupt_omega_cov = (self.zupt_omega_std**2)*self.g.new_ones(3)
        self.zupt_acc_cov = (self.zupt_acc_std**2)*self.g.new_ones(3)

        # each ZUPT IMU measurement outsides a threshold discards measumrent
        self.max_omega_norm = max_omega_norm
        self.max_acc_norm = max_acc_norm
        self.max_omega = max_omega
        self.max_acc = max_acc

        # number of increments for initialization
        self.N_init = N_init
        # frequency of rotation normalization
        self.N_normalize = N_normalize

        # skew symmetric matrix
        self.Wg = self.SO3.wedge(self.g)

    def forward(self, ts, us, zupts, covs):
        """Kalman filter loop"""
        dts = ts[1:] - ts[:-1]
        zupts = self.init(ts, us, zupts)
        for i in range(1, self.N):
            self.propagate(i, dts[i-1], us[i], zupts[i])
            self.update(i, us[i], covs[i], zupts[i])
            if i % self.N_normalize == 0:
                self.normalize_rotation_matrix(i)

    def init_cov(self):
        """Initialize P0 and Q"""
        Q = self.g.new_ones(12).diag()
        omega_std = Q.new([2e-3, 2e-3, 2e-3])
        acc_std = Q.new([7e-2, 7e-2, 7e-2])
        b_omega_std = Q.new([5e-5, 5e-5, 5e-5])
        b_acc_std = Q.new([4e-5, 4e-5, 4e-5])

        Q[:3, :3] *= omega_std**2
        Q[3:6, 3:6] *= acc_std**2
        Q[6:9, 6:9] *= b_omega_std**2
        Q[9:12, 9:12] *= b_acc_std**2

        Rot0_std = 1e-0
        b_omega0_std = 1e-3
        b_acc0_std = 2e-2
        P0 = Q.new_zeros(15, 15)
        tmp = Q.new_ones(3).diag()
        P0[:2, :2] = Rot0_std**2 * Q.new_ones(2).diag()
        P0[9:12, 9:12] = b_omega0_std**2*tmp
        P0[12:15, 12:15] = b_acc0_std**2*tmp
        return P0, Q

    def init(self, ts, us, zupts):
        zupts = self.init_zupt(zupts)
        self.init_bias(ts, us, zupts)
        self.init_trajectory(ts, us)
        return zupts

    def init_zupt(self, zupts):
        """Do thresholding on ZUPT"""
        zupts[zupts > self.th_max_zupt] = 1
        zupts[zupts <= self.th_min_zupt] = 0
        zupts[:self.N_init] = 1 # start by stop
        return zupts

    def init_trajectory(self, ts, us):
        self.N = us.shape[0] # trajectory length
        # init trajectory to zero
        self.v = us.new_zeros(3)
        self.ps = us.new_zeros(self.N, 3)

    def init_bias(self, ts, us, zupts):
        """Init IMU bias and orientation with first measurements"""
        if zupts[:self.N_init].sum() == self.N_init:
            N_init = self.N_init
        else:
            N_init = torch.where(zupts[:self.N_init] == 0)[0][0].item()
            cprint('Bias initialized with only {} samples'.format(N_init),
                'yellow')
        u = us[:N_init].mean(dim=0)
        gravity = -u[3:6]
        self.Rot = self.SO3.from_2vectors(gravity, self.g)
        self.b_omega = u[:3]
        self.b_acc = self.Rot.t().mv(self.g) - gravity

        self.P = self.P0.clone()
        # init covariance
        H = u.new_zeros(6, 15)
        H[:3, 9:12] = self.Id3
        H[3:6, :3] = self.Rot.t().mm(self.Wg)
        H[3:6, 12:15] = -self.Id3
        R = self.Q[:6, :6]
        for i in range(N_init):
            S = axat(H, self.P) + R
            Kt, _ = torch.solve(self.P.mm(H.t()).t(), S)
            K = Kt.t()
            I_KH = self.IdP - K.mm(H)
            self.P = axat(I_KH, self.P.clone()) + axat(K, R)
        self.P = (self.P + self.P.t()).clone()/2

    def propagate(self, i, dt, u, zupt):
        self.propagate_cov(i, dt, u, zupt)
        z = 1 - zupt
        acc = z*(self.Rot.mv(u[3:6] - self.b_acc) + self.g)
        self.Rot = self.Rot.mm(self.SO3.exp(z*(u[:3] - self.b_omega)*dt))
        self.ps[i] = self.ps[i-1] + z*self.v*dt + 1/2*acc*(dt**2)
        self.v = self.v + acc*dt

    def propagate_cov(self, i, dt, u, zupt):
        F = self.F
        G = self.G
        Rot = self.Rot

        z = 1 - zupt
        F[3:6, :3] = z*self.Wg
        F[3:6, 12:15] = -z*Rot
        F[6:9, 3:6] = z*self.Id3
        G[3:6, 3:6] = z*Rot

        v_skew_rot = self.SO3.wedge(self.v).mm(Rot)
        p_skew_rot = self.SO3.wedge(self.ps[i-1]).mm(Rot)
        tmp = z*torch.cat((Rot, v_skew_rot, p_skew_rot))
        G[:9, :3] = tmp
        F[:9, 9:12] = -tmp

        Phi = self.IdP + F*dt + 1/2*F.mm(F)*(dt**2)
        P = axat(Phi, self.P + axat(G*dt, self.Q))
        self.P = (P + P.t())/2

    def update(self, i, u, cov, zupt):
        H = self.H
        if zupt == 1:
            z = 1
        else:
            z = 0

        H[:3, 3:6] = self.Rot.t()
        H[0, 3:6] *= z

        self.r = torch.cat((- self.Rot.t().mv(self.v),
            u[:3] - self.b_omega,
            u[3:6] - self.b_acc + self.Rot.t().mv(self.g)))
        self.r[0] *= z

        z *= self.r[3:6].norm() < self.max_omega_norm
        z *= self.r[3:6].abs().max() < self.max_omega
        z *= self.r[6:9].norm() < self.max_acc_norm
        z *= self.r[6:9].abs().max() < self.max_acc
        self.r[3:9] *= z

        H[3:6, 9:12] = z*self.Id3
        H[6:9, 12:15] = z*self.Id3
        H[6:9, :3] = -z*self.Rot.t().mm(self.Wg)

        R = torch.diag(torch.cat((cov,
            self.zupt_omega_cov,
            self.zupt_acc_cov), 0))
        S = axat(H, self.P) + R
        Kt, _ = torch.solve(self.P.mm(H.t()).t(), S)
        K = Kt.t()
        self.xi = K.mv(self.r)
        self.state_update(i)
        self.covariance_update(i, K, H, R)

    def state_update(self, i):
        Rot, Xi = self.SE3.exp2(self.xi[:9])
        self.Rot = Rot.mm(self.Rot)
        self.v = Rot.mv(self.v) + Xi[:, 0]
        self.ps[i] = Rot.mv(self.ps[i].clone()) + Xi[:, 1]
        self.b_omega += self.xi[9:12]
        self.b_acc += self.xi[12:15]

    def covariance_update(self, i, K, H, R):
        I_KH = self.IdP - K.mm(H)
        P = axat(I_KH, self.P) + axat(K, R)
        self.P = (P + P.t())/2

    def normalize_rotation_matrix(self, i):
        U, _, V = torch.svd(self.Rot)
        S = self.Id3.clone()
        S[2, 2] = torch.det(U) * torch.det(V)
        self.Rot = U.mm(S).mm(V.t())

    def nets2iekf(self, net, us, Nshift, zupts):
        """Nets input to KF input"""
        us = us.unsqueeze(0).float().cuda()

        net = net.cuda().float()
        with torch.no_grad():
            covs = net(us).squeeze()
        net = net.cpu()

        where_zupts = (zupts >= self.th_max_zupt).double()
        tmp = zupts.new_zeros(zupts.shape[0])
        tmp[499:] = torch.nn.functional.conv1d(
            where_zupts.t().unsqueeze(0),
            zupts.new_ones(1, 1, 500).double()).squeeze()
        tmp = tmp >= 500

        us = us.squeeze()
        us = us[Nshift:].double().cpu()
        covs = covs[Nshift:].double().cpu()
        zupts = zupts[Nshift:].double().cpu()
        return us, zupts, covs

    class SO3:
        # tolerance criterion
        TOL = 1e-8
        Id = torch.eye(3)

        @staticmethod
        def outer(a, b):
            return torch.einsum('i, j -> ij', a, b)

        @classmethod
        def exp(cls, phi):
            angle = phi.norm()
            if angle < cls.TOL:
                return cls.Id + cls.wedge(phi)

            axis = phi / angle
            c = angle.cos()
            s = angle.sin()
            Rot = c*cls.Id + (1-c)*cls.outer(axis, axis) + s*cls.wedge(axis)
            return Rot

        @classmethod
        def inv_left_jacobian(cls, phi):
            angle = phi.norm()
            if angle < cls.TOL:
                return cls.Id - 1/2 * cls.wedge(phi)

            axis = phi/angle
            half_angle = angle/2
            cot = 1/half_angle.tan()

            J = half_angle*cot*cls.Id + (1-half_angle*cot)*cls.outer(axis, axis) - \
                half_angle*cls.wedge(axis)
            return J

        @classmethod
        def left_jacobian(cls, phi):
            angle = phi.norm()
            if angle < cls.TOL:
                return cls.Id + 1/2*cls.wedge(phi)

            axis = phi / angle
            s = angle.sin()
            c = angle.cos()

            J = (s/angle)*cls.Id + (1-s/angle)*cls.outer(axis, axis) +\
                ((1-c)/angle) * cls.wedge(axis)
            return J

        @classmethod
        def log(cls, Rot):
            cos_angle = (0.5 * Rot.trace() - 0.5).clamp(-1., 1.)
            # Clip cos(angle) to its proper domain to avoid NaNs from rounding
            # errors
            angle = cos_angle.acos()
            if angle < cls.TOL:
                return cls.vee(Rot - cls.Id)

            angle = angle.unsqueeze(0).unsqueeze(1)
            return cls.vee((0.5 * angle/angle.sin())*(Rot-Rot.t()))

        @staticmethod
        def vee(Phi):
            return torch.stack((Phi[2, 1],
                                Phi[0, 2],
                                Phi[1, 0]), dim=0)

        @staticmethod
        def wedge(phi):
            return phi.new([[0., -phi[2], phi[1]],
                            [phi[2], 0., -phi[0]],
                            [-phi[1], phi[0], 0.]])

        @classmethod
        def from_rpy(cls, roll, pitch, yaw):
            return cls.rotz(yaw).mm(cls.roty(pitch).mm(cls.rotx(roll)))

        @classmethod
        def rotx(cls, angle_in_radians):
            c = angle_in_radians.cos()
            s = angle_in_radians.sin()
            mat = c.new_zeros((3, 3))
            mat[0, 0] = 1
            mat[1, 1] = c
            mat[2, 2] = c
            mat[1, 2] = -s
            mat[2, 1] = s
            return mat

        @classmethod
        def roty(cls, angle_in_radians):
            c = angle_in_radians.cos()
            s = angle_in_radians.sin()
            mat = c.new_zeros((3, 3))
            mat[1, 1] = 1
            mat[0, 0] = c
            mat[2, 2] = c
            mat[0, 2] = s
            mat[2, 0] = -s
            return mat

        @classmethod
        def rotz(cls, angle_in_radians):
            c = angle_in_radians.cos()
            s = angle_in_radians.sin()
            mat = c.new_zeros((3, 3))
            mat[2, 2] = 1
            mat[0, 0] = c
            mat[1, 1] = c
            mat[0, 1] = -s
            mat[1, 0] = s
            return mat

        @classmethod
        def isclose(cls, x, y):
            return (x-y).abs() < cls.TOL

        @classmethod
        def to_rpy(cls, Rot):
            """Convert a rotation matrix to RPY Euler angles."""

            pitch = torch.atan2(-Rot[2, 0],
                                torch.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))
            near_pi_over_two_mask = cls.isclose(pitch, np.pi / 2.)
            near_neg_pi_over_two_mask = cls.isclose(pitch, -np.pi / 2.)

            if near_pi_over_two_mask:
                yaw = 0.
                roll = torch.atan2(Rot[0, 1], Rot[1, 1])
            else:
                if near_neg_pi_over_two_mask:
                    yaw = 0.
                    roll = -torch.atan2(Rot[0, 1], Rot[1, 1])
                else:
                    sec_pitch = 1. / pitch.cos()
                    yaw = torch.atan2(Rot[1, 0] * sec_pitch,
                                      Rot[0, 0] * sec_pitch)
                    roll = torch.atan2(Rot[2, 1] * sec_pitch,
                                       Rot[2, 2] * sec_pitch)
                    rpys = torch.stack([roll, pitch, yaw])

            return rpys

        @classmethod
        def from_2vectors(cls, v1, v2):
            """ Returns a Rotation matrix between vectors 'v1' and 'v2'    """
            v1 = v1/v1.norm()
            v2 = v2/v2.norm()
            v = torch.cross(v1, v2)
            cosang = (v1*v2).sum()
            sinang = v.norm()
            W = cls.wedge(v)
            Rot = cls.Id + W + (1-cosang)/(sinang**2)*W.mm(W)
            return Rot

        @classmethod
        def from_quaternion(cls, quat, ordering='wxyz'):
            """Form a rotation matrix from a unit length quaternion.
            Valid orderings are 'xyzw' and 'wxyz'.
            """

            if ordering is 'xyzw':
                qx = quat[0]
                qy = quat[1]
                qz = quat[2]
                qw = quat[3]
            elif ordering is 'wxyz':
                qw = quat[0]
                qx = quat[1]
                qy = quat[2]
                qz = quat[3]

            # Form the matrix
            mat = quat.new_empty(3, 3)

            qx2 = qx * qx
            qy2 = qy * qy
            qz2 = qz * qz

            mat[0, 0] = 1. - 2. * (qy2 + qz2)
            mat[0, 1] = 2. * (qx * qy - qw * qz)
            mat[0, 2] = 2. * (qw * qy + qx * qz)

            mat[1, 0] = 2. * (qw * qz + qx * qy)
            mat[1, 1] = 1. - 2. * (qx2 + qz2)
            mat[1, 2] = 2. * (qy * qz - qw * qx)

            mat[2, 0] = 2. * (qx * qz - qw * qy)
            mat[2, 1] = 2. * (qw * qx + qy * qz)
            mat[2, 2] = 1. - 2. * (qx2 + qy2)
            return mat


    class SE3:
        Id4 = torch.eye(4)
        Id5 = torch.eye(5)
        ones = Id4.new([0, 0, 0, 1])

        @classmethod
        def exp(cls, xi):
            p = BaseIEKF.SO3.left_jacobian(xi[:3]).mv(xi[3:])
            Rot = BaseIEKF.SO3.exp(xi[:3])
            return Rot, p

        @classmethod
        def exp2(cls, xi):
            Rot = BaseIEKF.SO3.exp(xi[:3])
            V = torch.stack((xi[3:6], xi[6:9]), 1)
            Xi = BaseIEKF.SO3.left_jacobian(xi[:3]).mm(V)
            return Rot, Xi

        @classmethod
        def inv(cls, chi):
            p_inv = -chi[:3, :3].t().mv(chi[:3, 3]).unsqueeze(1)
            Rot_p_inv = torch.cat((chi[:3, :3].t(), p_inv), 1)
            return torch.cat((Rot_p_inv, cls.ones), 0)

        @classmethod
        def log(cls, chi):
            """Logarithm for SE(3)"""
            phi = BaseIEKF.SO3.log(chi[:3, :3])
            rho = BaseIEKF.SO3.inv_left_jacobian(phi).mv(chi[:3, 3])
            xi = torch.cat((phi, rho))
            return xi

        @classmethod
        def log2(cls, chi):
            """Logarithm for SE_2(3)"""
            phi = BaseIEKF.SO3.log(chi[:3, :3])
            tmp = BaseIEKF.SO3.inv_left_jacobian(phi).mm(chi[:3, 3:])
            xi = torch.cat((phi, tmp[:, 0], tmp[:, 1]), 0)
            return xi


class RecorderIEKF(BaseIEKF):

    def __init__(self, **params):
        super().__init__(**params)

    def init_trajectory(self, ts, us):
        super().init_trajectory(ts, us)
        self.Rots = us.new_zeros((self.N, 3, 3))
        self.vs = us.new_zeros(self.N, 3)
        self.b_omegas = us.new_zeros(self.N, 3)
        self.b_accs = us.new_zeros(self.N, 3)

        self.rs = us.new_zeros(self.N, 9)  # residual
        self.xis = us.new_zeros(self.N, 9)  # innovation
        self.Ps = us.new_zeros(self.N, 15, 15)

        self.Rots[0] = self.Rot
        self.b_omegas[0] = self.b_omega
        self.b_accs[0] = self.b_acc
        self.Ps[0] = self.P

    def covariance_update(self, i, K, H, R):
        super().covariance_update(i, K, H, R)
        self.save(i)

    def save(self, i):
        self.Rots[i] = self.Rot
        self.vs[i] = self.v
        self.b_omegas[i] = self.b_omega
        self.b_accs[i] = self.b_acc
        self.rs[i] = self.r[:9]
        self.xis[i] = self.xi[:9]
        self.Ps[i] = self.P

    def dump(self, address, seq, zupts, covs):
        # turn cov
        J = torch.eye(9).repeat(self.Ps.shape[0], 1, 1)
        J[:, 3:6, :3] = SO3.wedge(self.vs)
        J[:, 6:9, :3] = SO3.wedge(self.ps)
        #self.Ps = axat(J, self.Ps[:, :9, :9])
        path = os.path.join(address, seq, 'iekf.p')
        mondict = {
            'Rots': self.Rots,
            'vs': self.vs,
            'ps': self.ps,
            'b_omegas': self.b_omegas,
            'b_accs': self.b_accs,
            'rs': self.rs,
            'Ps': self.Ps.diagonal(dim1=1, dim2=2),
            'zupts': zupts,
            'covs': covs,
        }
        for k, v in mondict.items():
            mondict[k] = v.float().detach().cpu()
        pdump(mondict, path)






