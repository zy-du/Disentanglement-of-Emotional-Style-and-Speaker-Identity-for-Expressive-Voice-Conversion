
import os
from os.path import join
from glob import glob
import numpy as np


data_dir = "./data"
mode = ['train', 'valid']
all_speakers = os.listdir("./data/train/lf0")

lf0_valid_dir = data_dir  + '/' + mode[1] + '/' + 'lf0'
for j in range(len(all_speakers)):
    speaker = all_speakers[j]
    lf0s = []
    lf0_VS_dir = lf0_valid_dir + '/' + speaker
    lf0_V_paths = sorted(glob(join(lf0_VS_dir, '*.npy')))
    lf0_V_paths = [item.replace('\\', '/') for item in lf0_V_paths]
    lf0_path = []
    lf0 = []


    for k in range(len(lf0_V_paths)):
        lf0_path = lf0_V_paths[k]
        lf0 = np.load(lf0_path)
        lf0s.append(lf0)

    lf0_all = np.concatenate(lf0s)
    nonzeros_indices = np.nonzero(lf0_all)
    log_f0s_mean, log_f0s_std = np.mean(lf0_all[nonzeros_indices]), np.std(lf0_all[nonzeros_indices])


    #out_path =out_dir + "/lf0_v/"
    out_path = os.path.join(lf0_VS_dir, speaker + '_stats.npz')
    out_path = out_path.replace('\\', '/')

    np.savez(out_path,
             log_f0s_mean=log_f0s_mean,
             log_f0s_std=log_f0s_std)
                 