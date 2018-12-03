# CellVision
Imaging single molecules in living cells with fluorescence microscopy offers unparalleled datasets that we use to crack unsolved problems in cell biology. Unfortunately, high light doses necessary to acquire high resolution images are harmful to cells, which severely limits our ability to visualize biological processes over long periods of time. We want to use deep learning to dramatically lower the light doses used in fluorescence microscopy while retaining high signal-to-noise ratio. This will allow us to perform experiments that are currently impossible.

# Steps for using csbdeep

Make sure to include this in your .bashrc

export PATH=/gpfs/share/skynet/apps/anaconda3/bin:$PATH

### 1. ssh to skynet
ssh <netid>@skynet.nyumc.org

### 2. activate tensorflow environment

source activate tensorflow-env

source /opt/DL/tensorflow/bin/tensorflow-activate

source /opt/DL/tensorboard/bin/tensorboard-activate

To Activate Pytorch

source activate pytorch-env

source /opt/DL/pytorch/bin/pytorch-activate

### 3. vi submit-jupyter.sh (only need to do this the first time)
pase the following contents
https://docs.google.com/document/d/1vAHun95oxFvRzQLtTvMPEXe_zUnt5o_ywUFZERazolA/edit

### 3. how to run submit-jupyter.sh:
bsub -Is -gpu "num=2:mode=exclusive_process:mps=yes" bash submit-jupyter.sh

### 4. An ssh tunnel command to skygpu appears in output of the command terminal
* The command looks like "ssh -N -L {port}:skygpu11:{port} <netid>@skynet.nyumc.org"
* Paste this command into a new terminal.

### 5. copy and paste address from first terminal window into broswer
* change skygpu to localhost, i.e.:
* http://localhost:9770/?token=<some token>

# Misc. Notes

In order to make folders/files accessible to everyone, run the following
* chown :lionnetlab path/to/dir

Run notebook without GPUs:
* bsub -Is bash submit-jupyter.sh

Example for bsub python submission:

* bsub -Is -gpu "num=1:mode=exclusive_process:mps=yes" python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

* bsub -Is -gpu "num=1:mode=exclusive_process:mps=yes" python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --gpu 0 --display_id 0

