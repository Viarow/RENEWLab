import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot_lib import scatter_ser_snr

suffix = ['65_65', '63_60', '60_57', '57_54', '54_51']
snr_dir = './results/LOS_64x2_16QAM/zf/'
ser_zf_dir = './results/LOS_64x2_16QAM/zf/'
# ser_langevine_dir = './results/LOS_64x2_16QAM/langevine'
useful_frames = range(500, 600)
save_dir = './results/LOS_64x2_16QAM/'

snr_list = []
for sf in suffix:
    snr_list.append(np.load(snr_dir + 'TxGains_' + sf + '_snr.npy'))
snr_array = np.concatenate(snr_list)

ser_zf_list = []
for sf in suffix:
    data = np.load(snr_dir + 'TxGains_' + sf + '_evm_ser.npy')
    ser_zf_list.append(data[2])
ser_zf_array = np.concatenate(ser_zf_list)

scatter_ser_snr(ser_zf_array, snr_array, save_dir)