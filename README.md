# RINS-W: Robust Inertial Navigation System on Wheels


## Overview

This repo contains a real-time approach for inertial navigation based only on an Inertial MeasurementUnit  (IMU)  for  self-localizing  wheeled  robots.  The  approach builds  upon  two  components: 1) a robust detector that uses deep neural networks to dynamically detects zero velocity; and 2) a state-of-the-art Kalman filter which incorporates this knowledge along with no lateral slip and vertical velocity as pseudo-measurements  for  localization.

## Code
Our implementation is done in Python a [Pytorch](https://pytorch.org/) for the adapter block of the system. The code was tested under Python 3.5.
 
### Installation & Prerequies
1.  Install [pytorch](http://pytorch.org). We perform all training and testing on its 1.5 version.
    
2.  Install required packages, e.g. with the pip3 command
```
pip3 install requirements.txt
```
    
4.  Clone this repo
```
git clone https://github.com/mbrossar/RINS-W.git
```

### Testing
Coming soon.

### Training
Coming soon.


## Papers
This repo is mainly based on the paper "RINS-W: Robust Inertial Navigation System on Wheels",  _International Conference on Intelligent Robots and Systems (IROS)_, 2019 [[IEEE paper](https://ieeexplore.ieee.org/document/8968593), [ArXiv paper](https://arxiv.org/pdf/1903.02210.pdf)]. The main differences with the paper are
- deep neural networks only estimates when zero velocity happens.
- deep neural networks is based on dilated convolutions and CNNs, which are much faster to train.
- the covariance of pseudo-measurement may depends on IMU inputs.



You can also see also the paper "AI-IMU Dead-Reckoning,"  _IEEE Transactions on Intelligent Vehicles_, 2020 [[IEEE paper](https://ieeexplore.ieee.org/document/9035481), [ArXiv paper](https://arxiv.org/pdf/1904.06064.pdf)].


### Citation

If you use this code in your research, please cite:

```
@INPROCEEDINGS{brossard2019,
  author={M. {Brossard} and A. {Barrau} and S. {Bonnabel}},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={{RINS-W: Robust Inertial Navigation System on Wheels}}, 
  year={2019},
  volume={},
  number={},
  pages={2068-2075},
  }

```

### Authors
Martin Brossard*, Axel Barrau* and Silvère Bonnabel*

*MINES ParisTech, PSL Research University, Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France


