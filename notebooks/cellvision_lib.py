import numpy as np
import pandas as pd
import os
from collections import defaultdict

def train_test_val(folder_path, channels = 1, train_pp = .6, test_pp = .2, val_pp = .2, set_seed = 1):
    #check to make sure percentages equal 1
    if train_pp+test_pp+val_pp != 1:
        raise Exception("train, test, val percentages must equal 1")
    
    #check to make sure channels in 1-5
    for c in list((channels,6)):
        if c not in list(range(1,7)):
            raise Exception("channels can only be in range 1-5")
    
    #Find file paths
    for root, dirs, files in os.walk(folder_path):
        root_path = root
        root_dir = dirs
        root_files = files
    
    #split each file to find relevant numbers
    sample_zplanes_folder = []
    for file in root_files:
        s,c,z = file.split('_')
        s_n = s.strip('sample')
        c_n = c.strip('channel')
        z_n = z.strip('z').split('.')[0]
        #create new entries where first part of key is sample #
        #second part of key is zplane #
        entry = list([str(s_n)+str('_')+str(z_n), file])
        sample_zplanes_folder.append(entry)
    
    #create dictionary with new keys for sample # and zplane #
    d = defaultdict(list)
    for key, entry in sample_zplanes_folder:
        d[key].append(entry)
    
    #full dictionary of files with corresponding sample/zplane #
    samples = d
    #just sample/zplane keys used to split data
    samples_list = list(samples.keys())


    #set number entries base on pp's for train, test, val
    train_p, test_p, val_p = round((len(samples_list))*train_pp), \
                             round((len(samples_list))*test_pp), \
                             round((len(samples_list))*val_pp)

    #set train seed
    np.random.seed(set_seed)
    #select training set
    train = list(np.random.choice(samples_list, size=train_p, replace=False))
    #remove training set from original list
    samples_list = list(set(samples_list) - set(train))

    #set test seed
    np.random.seed(set_seed)
    #select training set
    test = list(np.random.choice(samples_list, size=test_p, replace=False))
    #remove training set from original list
    val = list(set(samples_list) - set(test))
    
    #set channel list
    channel_set = list((channels,6))
    
    def finalize_paths(split_set, samples):
        paths = []
        for t in split_set:
            entry = samples[t]
            paths.append(entry)

        final = []
        for file in paths:
            temp = []
            for channel_file in file:
                s,c,z = channel_file.split('_')
                c_n = c.strip('channel')
                for c in channel_set:
                    if c_n == str(c):
                        temp.append('{}/{}'.format(root_path, channel_file))
            final.append(sorted(temp))  
        return final
    
    train_final = finalize_paths(train, samples)
    test_final = finalize_paths(test, samples)
    val_final = finalize_paths(val, samples)

    return(train_final, test_final, val_final)