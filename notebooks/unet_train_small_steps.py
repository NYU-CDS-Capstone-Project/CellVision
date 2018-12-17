import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import skimage
from skimage import io
import os
import glob
import numpy as np
from skimage import exposure, measure
from skimage.transform import rotate
from skimage.measure import compare_ssim as ssim
import re
from torch.utils.data import Dataset
from cellvision_lib import get_model_data_splits
import sys
import pickle

RUN_CHANNEL = int(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dir_to_file_lists(directory):
    os.chdir(directory)
    input_list = []
    target_list = []
    all_tifs = glob.glob("*.tif")
    input_tifs = [file for file in all_tifs if '_channel1_' in file]
    input_tifs.sort()
    output_tifs = [file for file in all_tifs if '_channel6_' in file]
    output_tifs.sort()
    return(input_tifs, output_tifs)

train, test, val = get_model_data_splits('/gpfs/data/lionnetlab/cellvision/pilotdata/20181009-top50', 
                                      channel = RUN_CHANNEL, 
                                      train_pp = .67, 
                                      test_pp = .165, 
                                      val_pp = .165, 
                                      seed = 1)
'''
train_images = train[0:num_images]
'''

def image_to_matrix_dataset(file_list, augmentation=False):
    """
    funciton takes list of file names and returns list of matrices
    list will be 6 times as long since data is flipped + rotated too
    params:
    augmentation: augment original training data with flips and rotations. Returns 6 times as many images.
    """
    
    mat_list = []
    for file in file_list:
        orig = io.imread(file)
        mat_list.append(orig)
        
        if augmentation:
            #vertical flip
            vert_flip = orig[::-1]
            mat_list.append(vert_flip)

            #horizonal flip
            horiz_flip = np.flip(orig,1)
            mat_list.append(horiz_flip)

            #rotate 90 degrees
            rot_90 = rotate(orig, 90)
            mat_list.append(rot_90)

            #rotate 180 degrees
            rot_180 = rotate(orig, 180)
            mat_list.append(rot_180)

            #rotate 270 degrees
            rot_270 = rotate(orig, 270)
            mat_list.append(rot_270)
    return(mat_list)

# full dataset
train_input = image_to_matrix_dataset(list(x[0] for x in train))
train_target = image_to_matrix_dataset(list(x[1] for x in train))

val_input = image_to_matrix_dataset(list(x[0] for x in val))
val_target = image_to_matrix_dataset(list(x[1] for x in val))

test_input = image_to_matrix_dataset(list(x[0] for x in test))
test_target = image_to_matrix_dataset(list(x[1] for x in test))

class two_image_dataset(Dataset):
    
    def __init__(self, input_tifs_mats, output_tifs_mats):
        
        self.input_tifs_mats = input_tifs_mats
        self.output_tifs_mats = output_tifs_mats
        assert (len(self.input_tifs_mats) == len(self.output_tifs_mats))
    
    def __len__(self):
        return len(self.input_tifs_mats)
    
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        input_mat = self.input_tifs_mats[key]
        output_mat = self.output_tifs_mats[key]
        return [input_mat, output_mat]
def two_image_collate_func(batch):
    """
    function that returns input and target as tensors
    """
    input_list = []
    target_list = []
    for datum in batch:
        input_list.append(datum[0].astype(dtype = 'float32')/32768)
        target_list.append(datum[1].astype(dtype = 'float32')/32768)
    input_tensor = torch.from_numpy(np.array(input_list))
    target_tensor = torch.from_numpy(np.array(target_list))
    return [input_tensor, target_tensor]

BATCH_SIZE = 4


# train
train_dataset = two_image_dataset(train_input, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=two_image_collate_func,
                                           shuffle=False)
# val
val_dataset = two_image_dataset(val_input, val_target)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=two_image_collate_func,
                                           shuffle=False)
# test
test_dataset = two_image_dataset(test_input, test_target)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=two_image_collate_func,
                                           shuffle=False)

# unet parts here

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
# UNET arch here
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, 512, 512).to(device)
        #print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
    
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    returns average ssim for loader
    @param: loader - data loader for the dataset to test against
    """
    ssim_list = []
    nmse_list = []
    model.eval()
    for inputs, targets in loader:
        outputs = model(inputs)
        # to cpu
        cur_out = outputs.cpu()
        cur_tar = targets.cpu()
        # get ssim for each pair
        for i in range(outputs.shape[0]):
            sing_out = (cur_out.data.numpy()[i,0,:,:]*32768 // 1).astype(np.int16)
            sing_tar = (cur_tar.data.numpy()[i,:,:]*32768 // 1).astype(np.int16)
            cur_ssim = ssim(sing_tar, sing_out, data_range=sing_out.max() - sing_out.min())
            sing_out = np.array(sing_out, dtype='int64')
            sing_tar = np.array(sing_tar, dtype='int64')
            cur_nmse = (np.abs(np.square(sing_tar - sing_out))).mean(axis=None) / (np.square(sing_tar - 0)).mean(axis=None)
            ssim_list.append(cur_ssim)
            nmse_list.append(cur_nmse)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    ssim_std = np.std(ssim_list)
    nmse_avg = sum(nmse_list) / len(nmse_list)
    nmse_std = np.std(nmse_list)
    
    return (ssim_avg, ssim_std, nmse_avg, nmse_std)

model = UNet(1, 1)
model = model.to(device)

learning_rate = 0.0001
num_epochs = 20

# Criterion and Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

os.chdir('/home/cra354/small_steps')


filename = 'unet_' + str(RUN_CHANNEL) + '_streaming_output_small_steps.txt'

for epoch in range(num_epochs): 
    for i, (inputs, targets) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1,1,512,512).to(device))
        # Backward and optimize
        loss.backward()
        optimizer.step()
        if i > 0 and i % 500 == 0:
            print("Epoch: {}, Step : {}".format(epoch, i))
            print("Training Loss : {}".format(loss))
            val_ssim, val_ssim_std, nmse_avg, nmse_std = test_model(val_loader, model)
            print("Validation SSIM: {},Validation SSIM Standard Deviation: {} ".format(val_ssim, val_ssim_std))
            print("Validation MSE: {}, Validation MSE Standard Deviation: {}".format(nmse_avg, nmse_std))
            # write to file so can see streaming...
            file = open(filename, "a")
            file.write("Epoch: {}, Step : {} \n".format(epoch, i))
            file.write("Training Loss : {} \n".format(loss))
            file.write("Validation SSIM: {},Validation SSIM Standard Deviation: {} \n".format(val_ssim, val_ssim_std))
            file.write("Validation MSE: {}, Validation MSE Standard Deviation: {} \n".format(nmse_avg, nmse_std))
            file.close()
            # save model each step
            modelname = 'model_' + str(RUN_CHANNEL)  +'_small_steps.p'
            pickle.dump(model, open(modelname, "wb" ))

