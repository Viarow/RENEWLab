import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot_lib import scatter_ser_snr

suffix = ['65_65', '63_60', '60_57', '57_54', '54_51', '51_48']
snr_dir = './results/LOS_64x2_QPSK/zf/'
ser_zf_dir = './results/LOS_64x2_QPSK/zf/'
# ser_langevine_dir = './results/LOS_64x2_16QAM/langevine'
# useful_frames = range(500, 600)
save_dir = './results/LOS_64x2_QPSK/'
num_users = 2

snr_list = np.zeros((len(suffix), ))
ser_zf_list = np.zeros((len(suffix), num_users))
for i, sf in enumerate(suffix):
    raw_snr = np.load(snr_dir + 'TxGains_' + sf + '_snr.npy')
    num_frames = raw_snr.shape[0]
    useful_frames = []
    for j in range(num_frames):
        if raw_snr[j] > 0:
            useful_frames.append(j)
    snr_list[i] = np.mean(raw_snr[useful_frames])

    raw_ser = np.load(snr_dir + 'TxGains_' + sf + '_evm_ser.npy')[2]
    ser_zf_array = np.mean(raw_ser[useful_frames], axis=0)
    ser_zf_list[i] = ser_zf_array

scatter_ser_snr(ser_zf_list, snr_list, save_dir)