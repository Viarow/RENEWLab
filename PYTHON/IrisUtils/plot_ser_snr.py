import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot_lib import scatter_ser_snr, plot_ser_evm

suffix = ['65_65', '63_60', '60_57', '57_54', '54_51', '51_48']
snr_dir = './results/LOS_64x2_16QAM/zf/'
ser_zf_dir = './results/LOS_64x2_16QAM/zf/'
ser_lang_dir = './results/LOS_64x2_16QAM/langevine_traj2/'
# useful_frames = range(500, 600)
save_dir = './results/LOS_64x2_16QAM/'
num_users = 2

snr_list = np.zeros((len(suffix), ))
ser_list = np.zeros((len(suffix), num_users * 2))
for i, sf in enumerate(suffix):
    raw_snr = np.load(snr_dir + 'TxGains_' + sf + '_snr.npy')
    num_frames = raw_snr.shape[0]
    useful_frames = []
    for j in range(num_frames):
        if raw_snr[j] > 0:
            useful_frames.append(j)
    snr_list[i] = np.mean(raw_snr[useful_frames])

    raw_zf_ser = np.load(ser_zf_dir + 'TxGains_' + sf + '_evm_ser.npy')[2]
    ser_zf_array = np.mean(raw_zf_ser[useful_frames], axis=0)
    ser_list[i, 0:num_users] = ser_zf_array
    
    raw_lang_ser = np.load(ser_lang_dir + 'TxGains_' + sf + '_evm_ser.npy')[2]
    ser_lang_array = np.mean(raw_lang_ser[useful_frames], axis=0)
    ser_list[i, num_users:(2*num_users)] = ser_lang_array

colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red']
markers = ['o', 'v', 'o', 'v']
labels = ['zf user0', 'zf user1', 'lang user0', 'lang user1']

save_path = save_dir+'/ser_snr_traj2.png'
title = 'SER vs SNR, langevine 2 trajectories'
plot_ser_evm(ser_list, snr_list, colors, markers, labels, title, save_path)