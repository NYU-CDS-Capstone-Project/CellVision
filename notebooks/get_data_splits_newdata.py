import os
import glob
import random

def get_model_data_splits(imgs_dir, channel=1, train_pp=.6, test_pp=.2, val_pp=.2, seed=1):
    """
    Function which returns the a list containing a tuple of the channel image and its
    matched reference channel for training, validation, and test.
    
    Expects filenames of the format: amr_#_sample#_channel#.tif
    
    returns train, val, test
    """
    REF_CHANNEL = 4
    random.seed(seed)
    #check to make sure percentages equal 1
    if train_pp+test_pp+val_pp != 1:
        raise Exception("train, test, val percentages must equal 1")
    
    #check to make sure channels in 1-4
    if channel > 4:
        raise Exception("channels can only be in range 1-4")
    
    img_paths = sorted(glob.glob(imgs_dir + '/*_channel{}.tif'.format(channel)))
    ref_paths = sorted(glob.glob(imgs_dir + '/*_channel{}.tif'.format(REF_CHANNEL)))
    random.shuffle(img_paths)
    sample_prefixes = {}
    for img_path in img_paths:
        fname = os.path.basename(img_path)
        end_sample_prefix_index = fname.find('_channel{}.tif'.format(channel)) 
        end_index = fname.find('.tif') 
        sample_prefix = fname[0:end_sample_prefix_index]
        ref_path = '{}/{}_channel{}.tif'.format(imgs_dir, sample_prefix, REF_CHANNEL)
        if ref_path not in ref_paths:
            print("WARNING! Missing ref for " + ref_path)
            continue
        if sample_prefix not in sample_prefixes:
            sample_prefixes[sample_prefix] = [ (img_path, ref_path) ]
        else:
            sample_prefixes[sample_prefix] = sample_prefixes[sample_prefix] + [(img_path, ref_path)]
    
    sample_prefixes_lst = sorted(sample_prefixes.keys())
    num_samples = len(sample_prefixes_lst)
    random.shuffle(sample_prefixes_lst)
    
    train_end_idx = round(num_samples*train_pp)
    val_end_idx = train_end_idx + round(num_samples*val_pp)
    
    train_samples = sample_prefixes_lst[0:train_end_idx]
    val_samples = sample_prefixes_lst[train_end_idx:val_end_idx]
    test_samples = sample_prefixes_lst[val_end_idx:]
    
    train = []
    for sample in train_samples:
        train = train + sample_prefixes[sample]
        
    val = []
    for sample in val_samples:
        val = val + sample_prefixes[sample]
        
    test = []
    for sample in test_samples:
        test = test + sample_prefixes[sample]
        
    return train, val, test