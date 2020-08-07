
import torch
import time
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from termcolor import cprint
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt, pltt, plts, axat, pltt, plts
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3
from src.iekf import RecorderIEKF as IEKF
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score


class BaseProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        self.res_dir = res_dir
        self.tb_dir = tb_dir
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt # (s)
        self.address, self.tb_address = self.find_address(address)
        if address is None:  # create new address
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
        else:  # pick the network parameters
            self.net_params = pload(self.address, 'net_params.p')
            self.train_params = pload(self.address, 'train_params.p')
            self._ready = True
        self.path_weights = os.path.join(self.address, 'weights.pt')
        self.net = self.net_class(**self.net_params)
        if self._ready:  # fill network parameters
            self.load_weights()
        self.seq = None

    def find_address(self, address):
        """return path where net and training info are saved"""
        if address == 'last':
            addresses = sorted(os.listdir(self.res_dir))
            tb_address = os.path.join(self.tb_dir, str(len(addresses)))
            address = os.path.join(self.res_dir, addresses[-1])
        elif address is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            address = os.path.join(self.res_dir, now)
            mkdir(address)
            tb_address = os.path.join(self.tb_dir, now)
        else:
            tb_address = None
        return address, tb_address

    def load_weights(self):
        weights = torch.load(self.path_weights)
        self.net.load_state_dict(weights)
        self.net.cuda()

    def train(self, dataset_class, dataset_params, train_params):
        """train the neural network. GPU is assumed"""
        self.train_params = train_params
        pdump(self.train_params, self.address, 'train_params.p')
        ydump(self.train_params, self.address, 'train_params.yaml')

        hparams = self.get_hparams(dataset_class, dataset_params, train_params)
        ydump(hparams, self.address, 'hparams.yaml')

        # define datasets
        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        # get class
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # define optimizer, scheduler and loss
        dataloader = DataLoader(dataset_train, **dataloader_params)
        optimizer = Optimizer(self.net.parameters(), **optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)
        criterion = Loss(**loss_params)

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        # init net w.r.t dataset
        self.net = self.net.cuda()
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u)

        # start tensorboard writer
        writer = SummaryWriter(self.tb_address)
        start_time = time.time()
        best_loss = torch.Tensor([float('Inf')])

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(
                epoch, loss_epoch.item()))
            scheduler.step(epoch)

        def write_time(epoch, start_time):
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " +
                "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            if loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        # training loop !
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            write(epoch, loss_epoch)
            scheduler.step(epoch)
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')
        self.load_weights()
        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
            }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            us = dataloader.dataset.add_noise(us.cuda())
            hat_xs = self.net(us)
            loss = criterion(xs.cuda(), hat_xs)/len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                hat_xs = self.net(us.cuda().unsqueeze(0))
                loss = criterion(xs.cuda().unsqueeze(0), hat_xs)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.path_weights)
        self.net.train().cuda()

    def get_hparams(self, dataset_class, dataset_params, train_params):
        """return all training hyperparameters in a dict"""
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get training class parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params, modes):
        """test a network once training is over"""

        # get loss function
        Loss = self.train_params['loss_class']
        loss_params = self.train_params['loss']
        criterion = Loss(**loss_params)

        # test on each type of sequence
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset, criterion)
            self.display_test(dataset_class, dataset_params, mode)

    def loop_test(self, dataset, criterion):
        """Forward loop over test data"""
        self.net.eval()
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            us, xs = dataset[i]
            with torch.no_grad():
                hat_xs = self.net(us.cuda().unsqueeze(0))
            loss = criterion(xs.cuda().unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self, dataset_class, dataset_params, mode):
        raise NotImplementedError
    
    def get_results(self, seq):
        return pload(self.address, seq, 'results.p')['hat_xs']
    


    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
        else:
            axs.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(self.address, self.seq, name + '.png'))
        plt.close('all')


class ZUPTProcessing(BaseProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)

    def display_test(self, dataset_class, dataset_params, mode):
        dataset = dataset_class(**dataset_params, mode=mode)
        zupts = torch.zeros(0)
        hat_zupts = torch.zeros(0)

        for i, seq in enumerate(dataset.sequences):
            print('\n', '---------  Result for sequence ' + seq + '  ---------')
            self.seq = seq
            # get ground truth pose
            self.gt = dataset.load_gt(i)

            # get data and estimate
            self.us, self.zupt = dataset[i]
            self.N = self.us.shape[0]
            self.hat_zupt = torch.sigmoid(self.get_results(seq))
            self.ts = torch.linspace(0, self.N*self.dt, self.N)

            self.convert()
            self.zupt_plot()
            zupts = torch.cat((zupts, self.zupt))
            hat_zupts = torch.cat((hat_zupts, self.hat_zupt))
            
        zupts = zupts.numpy()
        hat_zupts = hat_zupts.numpy()
        fpr, tpr, ths = roc_curve(zupts, hat_zupts)
        precision, recall, ths2 = precision_recall_curve(zupts, hat_zupts)
        auc = roc_auc_score(zupts, hat_zupts)

        self.print_and_save_auc(auc)
        self.roc_plot(fpr, tpr)
        self.pr_plot(precision, recall)

    

    def zupt_plot(self):
        title = "ROC curve " + self.end_title
        vs = self.gt['vs'].norm(dim=1)
        vs /= vs.max()
        zupt = 1 - self.zupt
        hat_zupt = 1 - self.hat_zupt
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t(s)$', ylabel='ZUPT',
            title=title)
        plt.plot(self.ts, vs, color="red", label=r'true speed')
        plt.plot(self.ts, zupt, color="black", label=r'true')
        plt.plot(self.ts, hat_zupt, color="blue", label=r'net')
        
        self.savefig(ax, fig, self.seq + "_zupt")
    
    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l
        
        # m/s -> km/h
        l = 3.6
        self.gt['vs'] *= l

    def roc_plot(self, fpr, tpr):
        title = "ROC curve"
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='false positive rate', ylabel='true positive rate',
            title=title)
        plt.plot(fpr, tpr, color="blue", label=r'net')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.savefig(ax, fig, 'roc')

    def pr_plot(self, precision, recall):
        title = "precision recall curve"
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='precision', ylabel='recall', title=title)
        plt.plot(recall, precision, color="blue", label=r'net')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.savefig(ax, fig, 'precisionrecall')
        
    def print_and_save_auc(self, auc):
        print('')
        print('Area Under Curve (AUC): {:.5f}'.format(auc))

        mondict = {
            "auc": auc.item(),
        }
        ydump(mondict, self.address, "net_result.yaml")



class KalmanProcessing(BaseProcessing):
    def __init__(self, res_dir, tb_dir, net_class, bbb_net_params, address, dt, iekf_params, train_params):
        super().__init__(res_dir, tb_dir, net_class, bbb_net_params, None, dt)
        # delete and replace address
        shutil.rmtree(self.address)
        self.address = address
        self.train_params = train_params
        self.iekf_params = iekf_params

    def loop_test(self, dataset, criterion):
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            print('Testing sequence ' + seq + ' (mode is ' + dataset.mode + ')')
            ts, us, Nshift = dataset.get_test(i)
            kf = IEKF(**self.iekf_params)
            zupts = torch.sigmoid(self.get_results(seq))
            us, zupts, covs = kf.nets2iekf(self.net, us, Nshift, zupts)
            # run filter !
            kf.forward(ts, us, zupts, covs)
            kf.dump(self.address, seq, zupts, covs)

    def display_test(self, dataset_class, dataset_params, mode):
        dataset = dataset_class(**dataset_params, mode=mode)
        for i, seq in enumerate(dataset.sequences):
            print('\n', '---------  Result for sequence ' + seq + '  ---------')
            self.seq = seq
            # get ground truth pose
            self.gt = dataset.load_gt(i)
            self.gt['Rots'] = SO3.from_quaternion(self.gt['qs'].cuda()).cpu()

            # get data and estimate
            self.us, self.zupt = dataset[i]
            self.iekf = self.get_iekf_results(seq)
            self.N = self.iekf['ps'].shape[0]
            N0 = self.us.shape[0]-self.N
            self.us = self.us[N0:]
            self.zupt = self.zupt[N0:]
            for key, val in self.gt.items():
                self.gt[key] = val[N0:]
            self.ts = torch.linspace(0, self.N*self.dt, self.N)
            
            self.align_traj()
            self.convert()
            self.plot_orientation()
            self.plot_velocity()
            self.plot_velocity_in_body_frame()
            self.plot_position()
            self.plot_horizontal_position()
            self.plot_bias_gyro()
            self.plot_bias_acc()
            self.plot_gyro()
            self.plot_acc()
            self.plot_zupt()
            self.plot_orientation_err()
            self.plot_velocity_err()
            self.plot_body_velocity_err()
            self.plot_position_err()

    def get_iekf_results(self, seq):
        return pload(self.address, seq, 'iekf.p')
    
    def align_traj(self):
        """yaw only and position alignment at initial time"""
        self.gt['rpys'] = SO3.to_rpy(self.gt['Rots'].cuda()).cpu()
        self.iekf['rpys'] = SO3.to_rpy(self.iekf['Rots'].cuda()).cpu()
        
        self.gt['ps'] -= self.gt['ps'][0].clone()
        self.iekf['ps'] -= self.iekf['ps'][0].clone()
        rpys = self.gt['rpys'][:2] - self.iekf['rpys'][:2]
        Rot = SO3.from_rpy(rpys[:, 0], rpys[:, 1], rpys[:, 2])
        Rot = Rot[0].repeat(self.iekf['ps'].shape[0], 1, 1)
        
        self.iekf['Rots'] = Rot.bmm(self.iekf['Rots'])
        self.iekf['vs'] = bmv(Rot, self.iekf['vs'])
        self.iekf['ps'] = bmv(Rot, self.iekf['ps'])
        self.iekf['rpys'] = SO3.to_rpy(self.iekf['Rots'].cuda()).cpu()

    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l
        
        # m/s -> km/h
        l = 3.6
        self.gt['vs'] *= l
        self.iekf['vs'] *= l
        self.iekf['Ps'][:, 3:6] *= l**2

        # rad/s -> deg/s
        l = 180/np.pi
        self.iekf['b_omegas'] *= l
        self.us[:, :3] *= l
        self.iekf['Ps'][:, 9:12] *= l**2
        
        # rad -> deg
        l = 180/np.pi
        self.gt['rpys'] *= l
        self.iekf['rpys'] *= l

    def plot_orientation(self):
        title = "Orientation as function of time " + self.end_title
        true = self.gt['rpys']
        mean = self.iekf['rpys']
        std = 180/np.pi*3*self.iekf['Ps'][:, :3].sqrt()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')
        
        for i in range(3):
            axs[i].plot(self.ts, true[:, i], color="black")
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'ground truth', r'IEKF', r'$3\sigma$'], ncol=3)
        self.savefig(axs, fig, 'orientation_time')
    
        
    def plot_velocity(self):
        title = "Velocity as function of time " + self.end_title
        true = self.gt['vs']
        mean = self.iekf['vs']
        std = 3*self.iekf['Ps'][:, 3:6].sqrt()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$\mathbf{v}_n^x$ (km/h)', title=title)
        axs[1].set(ylabel='$\mathbf{v}_n^y$ (km/h)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$\mathbf{v}_n^z$ (km/h)')
        
        for i in range(3):
            axs[i].plot(self.ts, true[:, i], color="black")
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'ground truth', r'IEKF', r'$3\sigma$'], ncol=3)
        self.savefig(axs, fig, 'velocity')
    
    def plot_velocity_in_body_frame(self):
        title = "Body velocity as function of time " + self.end_title
        true = bmv(self.gt['Rots'].transpose(1, 2), self.gt['vs'])
        mean = bmv(self.iekf['Rots'].transpose(1, 2), self.iekf['vs'])
        # get 3 sigma uncertainty
        P = torch.diag_embed(self.iekf['Ps'][:, :6], offset=0, dim1=-2, dim2=-1)
        J = P.new_zeros(P.shape[0], 3, 6)
        J[:, :, :3] = SO3.wedge(mean)
        J[:, :, 3:6] = self.iekf['Rots'].transpose(1, 2)
        std = J.bmm(P).bmm(J.transpose(1, 2)).diagonal(dim1=1, dim2=2).sqrt()

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$(\mathbf{R}_n^T\mathbf{v}_n)^x$ (km/h)', 
            title=title)
        axs[1].set(ylabel='$(\mathbf{R}_n^T\mathbf{v}_n)^y$ (km/h)')
        axs[2].set(xlabel='$t$ (min)',
            ylabel='$(\mathbf{R}_n^T\mathbf{v}_n)^z$ (km/h)')
        
        for i in range(3):
            axs[i].plot(self.ts, true[:, i], color="black")
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'ground truth', r'IEKF', r'$3\sigma$'], ncol=3)
        self.savefig(axs, fig, 'body_velocity')
    
    def plot_position(self):
        title = "Position as function of time " + self.end_title
        true = self.gt['ps']
        mean = self.iekf['ps']
        std = 3*self.iekf['Ps'][:, 6:9].sqrt()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$\mathbf{p}_n^x$ (km)', title=title)
        axs[1].set(ylabel='$\mathbf{p}_n^y$ (km)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$\mathbf{p}_n^z$ (km)')
        
        for i in range(3):
            axs[i].plot(self.ts, true[:, i], color="black")
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'ground truth', r'IEKF', r'$3\sigma$'], ncol=3)
        self.savefig(axs, fig, 'position_time')
    
    def plot_horizontal_position(self):
        title = "Horizontal position " + self.end_title
        true = self.gt['ps']
        mean = self.iekf['ps']
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=self.figsize)
        ax.set(ylabel='$\mathbf{p}_n^x$ (km)', label='$\mathbf{p}_n^y$ (km)', title=title)
        ax.plot(true[:, 0], true[:, 1], color="black")
        ax.plot(mean[:, 0], mean[:, 1], color="green")
        fig.legend([r'ground truth', r'IEKF'], ncol=2)
        self.savefig(ax, fig, 'horizontal_position')
    
    def plot_bias_gyro(self):
        title = "Gyro biases as function of time " + self.end_title
        mean = self.iekf['b_omegas']
        std = 3*self.iekf['Ps'][:, 9:12].sqrt()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$(\mathbf{b}_n^\omega)^x$ (deg/s)', title=title)
        axs[1].set(ylabel='$(\mathbf{b}_n^\omega)^y$ (deg/s)')
        axs[2].set(xlabel='$t$ (min)', 
                   ylabel='$(\mathbf{b}_n^\omega)^z$ (deg/s)')
        
        for i in range(3):
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IEKF'])
        self.savefig(axs, fig, 'bias_gyro')

    def plot_bias_acc(self):
        title = "Accelerometer biases as function of time " + self.end_title
        mean = self.iekf['b_accs']
        std = 3*self.iekf['Ps'][:, 9:12].sqrt()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$(\mathbf{b}_n^a)^x$ ($m/s^2$)', title=title)
        axs[1].set(ylabel='$(\mathbf{b}_n^a)^y$ ($m/s^2$)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$(\mathbf{b}_n^a)^z$ ($m/s^2$)')
        
        for i in range(3):
            axs[i].plot(self.ts, mean[:, i], color="green")
            axs[i].plot(self.ts, (mean+std)[:, i], color='green', alpha=0.5)
            axs[i].plot(self.ts, (mean-std)[:, i], color='green', alpha=0.5)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IEKF'])
        self.savefig(axs, fig, 'bias_acc')

    def plot_gyro(self):
        title = "Gyro as function of time " + self.end_title
        mean = self.us[:, :3]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel=r'$\boldsymbol{\omega}_n^x$ ($deg/s$)', title=title)
        axs[1].set(ylabel=r'$\boldsymbol{\omega}_n^y$ ($deg/s$)')
        axs[2].set(xlabel='$t$ (min)', ylabel=r'$\boldsymbol{\omega}_n^z$ ($deg/s$)')
        
        for i in range(3):
            axs[i].plot(self.ts, mean[:, i], color="blue")
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IMU'])
        self.savefig(axs, fig, 'gyro')

    def plot_acc(self):
        title = "Accelerometer as function of time " + self.end_title
        mean = self.us[:, 3:6]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel=r'$\mathbf{a}_n^x$ ($m/s^2$)', title=title)
        axs[1].set(ylabel=r'$\mathbf{a}_n^y$ ($m/s^2$)')
        axs[2].set(xlabel='$t$ (min)', ylabel=r'$\mathbf{a}_n^z$ ($m/s^2$)')
        
        for i in range(3):
            axs[i].plot(self.ts, mean[:, i], color="blue", label=r'IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'acc')

    def plot_zupt(self):
        pass

    def plot_covs(self):
        title = "Standard deviation measurement as function of time " + self.end_title
        std = self.iekf['covs'].sqrt().log()
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='log scale', title=title)
        axs[1].set(ylabel='log scale')
        axs[2].set(xlabel='$t$ (min)', ylabel='log scale')
        
        for i in range(5):
            axs[i].plot(self.ts, std[:, i])
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'position_error')

    def plot_orientation_err(self):
        title = "Position error as function of time " + self.end_title
        err = SO3.log(bmtm(self.gt['Rots'].cuda(), 
                        self.iekf['Rots'].cuda())).cpu()
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')
        
        for i in range(3):
            axs[i].plot(self.ts, err[:, i], color="blue")
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def plot_velocity_err(self):
        title = "Velocity error as function of time " + self.end_title
        err = self.gt['vs'] - self.iekf['vs']
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$\mathbf{v}_n^x$ (km/h)', title=title)
        axs[1].set(ylabel='$\mathbf{v}_n^y$ (km/h)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$\mathbf{v}_n^z$ (km/h)')
        
        for i in range(3):
            axs[i].plot(self.ts, err[:, i], color="blue")
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IEKF'])
        self.savefig(axs, fig, 'velocity_error')


    def plot_body_velocity_err(self):
        title = "Body velocity error as function of time " + self.end_title
        vs = bmv(self.gt['Rots'], self.gt['vs'])
        hat_vs = bmv(self.iekf['Rots'], self.iekf['vs'])
        err = vs - hat_vs
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$(\mathbf{R}_n^T\mathbf{v}_n)^x$ (km/h)', title=title)
        axs[1].set(ylabel='$\mathbf{v}_n^y$ (km/h)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$\mathbf{v}_n^z$ (km/h)')
        
        for i in range(3):
            axs[i].plot(self.ts, err[:, i], color="blue")
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IEKF'])
        self.savefig(axs, fig, 'body_velocity_error')

    def plot_position_err(self):
        title = "Position error as function of time " + self.end_title
        err = self.gt['ps'] - self.iekf['ps']
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='$\mathbf{p}_n^x$ (m)', title=title)
        axs[1].set(ylabel='$\mathbf{p}_n^y$ (m)')
        axs[2].set(xlabel='$t$ (min)', ylabel='$\mathbf{p}_n^z$ (m)')
        
        for i in range(3):
            axs[i].plot(self.ts, err[:, i], color="blue")
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        fig.legend([r'IEKF'])
        self.savefig(axs, fig, 'position_error')
