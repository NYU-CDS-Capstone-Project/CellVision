# CellVision
Imaging single molecules in living cells with fluorescence microscopy offers unparalleled datasets that we use to crack unsolved problems in cell biology. Unfortunately, high light doses necessary to acquire high resolution images are harmful to cells, which severely limits our ability to visualize biological processes over long periods of time. We want to use deep learning to dramatically lower the light doses used in fluorescence microscopy while retaining high signal-to-noise ratio. This will allow us to perform experiments that are currently impossible.

Make sure to include this in your .bashrc

export PATH=/gpfs/share/skynet/apps/anaconda3/bin:$PATH

# Steps for using csbdeep

# 1. ssh to skynet
ssh <netid>@skynet.nyumc.org

# 2. activate tensorflow environment

source activate tensorflow-env

source /opt/DL/tensorflow/bin/tensorflow-activate

source /opt/DL/tensorboard/bin/tensorboard-activate

# 3. run submit-jupyter.sh
bsub -Is -gpu "num=2:mode=exclusive_process:mps=yes" bash submit-jupyter.sh

# 4. ssh tunnel to skygpu in new terminal window
# copy what looks like this : ssh -N -L 9770:skygpu15:9770
<netid>@skynet.nyumc.org

# 5. copy and paste address from first terminal window into broswer
# change skygpu to localhost, i.e.:
# http://localhost:9770/?token=<some token>
