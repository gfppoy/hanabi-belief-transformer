# Training a Belief Over a Pool of Hanabi Policies
This codebase is largely based off https://github.com/facebookresearch/hanabi_SAD with some modifications. The code is set up to train a belief model
with the transformer architecture, and may be run via running two simultaneous programs: `python pyhanabi/train_multi.py`, which initializes and trains
a belief model, and `python pyhanabi/multi_replay.py`, which writes and tokenizes action-observation histories 
asynchronously ready for the belief model to read and train over. As written the belief model trains over policies `models/sad-op/M1.pthw, 
models/sad-op/M3.pthw, models/sad-op/M5.pthw, models/sad-op/M7.pthw, models/sad-op/M9.pthw, models/sad-op/M11.pthw` (see `lines 423-427` in 
`pyhanabi/train_multi.py`), and so modifying `lines 290, 423-427` in `pyhanabi/train_multi.py` will change the pool of policies the belief is trained over.

## Compile
We have been using `pytorch-1.5.1`, `cuda-10.1`, and `cudnn-v7.6.5` in our development environment.
Other settings may also work but we have not tested it extensively under different configurations.
We also use `conda/miniconda` to manage environments.
```bash
# create new conda env
conda create -n hanabi python=3.7
conda activate hanabi

# install pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install other dependencies
pip install numpy
pip install psutil

# if the current cmake version is < 3.15
conda install -c conda-forge cmake
```

### Clone & Build this repo
For convenience, add the following lines to your `.bashrc`,
after the line of `conda activate xxx`.

```bash
# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

Clone & build.
```bash
git clone --recursive https://github.com/facebookresearch/hanabi.git

cd hanabi
mkdir build
cd build
cmake ..
make -j10
```

## Run

`hanabi/pyhanabi/tools` contains some example scripts to launch training
runs. `dev.sh` is a fast lauching script for debugging. It needs 2 gpus to run,
1 for training and 1 for simulation. Other scripts are examples for a more formal
training run, they require 3 gpus, 1 for training and 2 for simulation.

The important flags are:

`--sad 1` to enable "Simplified Action Decoder";

`--pred_weight 0.25` to enable auxiliary task and multiply aux loss with 0.25;

`--shuffle_color 1` to enable other-play.

```bash
cd pyhanabi
sh tools/dev.sh
```

## Trained Models

Run the following command to download the trained models used to
produce tables in the paper.
```bash
cd model
sh download.sh
```
To evaluate a model, simply run
```bash
cd pyhanabi
python tools/eval_model.py --weight ../models/sad_2p_10.pthw --num_player 2
```

## Related Repos

The results on Hanabi can be further improved by running search on top
of our agents. Please refer to the [paper](https://arxiv.org/abs/1912.02318) and
[code](https://github.com/facebookresearch/Hanabi_SPARTA) for details.

We also open-sourced a single agent implementation of R2D2 tested on Atari
[here](https://github.com/facebookresearch/rela).

## Contribute

### Python
Use [`black`](https://github.com/psf/black) to format python code,
run `black *.py` before pushing

### C++
The root contains a `.clang-format` file that define the coding style of
this repo, run the following command before submitting PR or push
```bash
clang-format -i *.h
clang-format -i *.cc
```

## Copyright
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
