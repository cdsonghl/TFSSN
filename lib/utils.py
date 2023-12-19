import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, kaiser
import numpy as np

class SWJTU_Gear_Data:
    def __init__(self, filename, label_file_name, file_path='',fs = 12800,):
        self.file_path = file_path
        path = self.file_path +os.sep + filename
        path_label = self.file_path +os.sep + label_file_name
        self.df = pd.read_csv(path)
        self.df_label = pd.read_csv(path_label)
        self.fs = fs

    def plt_stft(self, data_id, save_path, fs_max):
        nperseg = 2048  # window size
        noverlap = int(nperseg * 0.75)  # overlap size
        nfft = nperseg  # FFT size
        frequencies_stft, times_stft, Zxx = stft(self.df[data_id], fs=self.fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,window=kaiser(nperseg, 15))
        mask = frequencies_stft <= fs_max
        dpi = 100
        l_t = len(times_stft) / dpi
        l_f = len(frequencies_stft[mask]) / dpi
        plt.figure(figsize=(l_t, l_f))
        plt.pcolormesh(times_stft, frequencies_stft[mask], 10 * np.log10(np.abs(Zxx[mask, :])), shading='gouraud',
                       cmap='jet')
        plt.ylim([frequencies_stft[mask][0], frequencies_stft[mask][-1]])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # plt.show(bbox_inches='tight')
        plt.savefig(save_path+'/{}/{}.png'.format(str(int(self.df_label[data_id])), data_id), dpi=dpi)
        plt.close()
        plt.clf()

        return int(l_t * dpi), int(l_f * dpi)
def get_fs(teeth_num_list, speed):
    fs_list = []
    for i in range(0, len(teeth_num_list),2):
        fs = round(teeth_num_list[i] * speed)
        intermediate_shaft_speed = speed * teeth_num_list[i] / teeth_num_list[i+1]
        speed = intermediate_shaft_speed
        fs_list.append(fs)
    return fs_list

def get_split_coordinate(fs_list, fs_max, l_f):
    T_list = []
    H_list = []
    fs_id_list = []
    for fs in fs_list:
        T = l_f - math.floor((fs + math.floor(fs / (10 ** math.floor(math.log10(fs)))) * (10 ** (math.floor(math.log10(fs))-1))) / fs_max * l_f)
        T_list.append(T)
        H = math.floor(2 * math.floor(fs / (10 ** math.floor(math.log10(fs)))) * (10 ** (math.floor(math.log10(fs))-1)) / fs_max * l_f)
        H_list.append(H)
    for n in range(len(fs_list)):
        fs_id = 'fs_' + str(n)
        fs_id_list.append(fs_id)
    fs_dict = dict(zip(fs_id_list,list(zip(T_list, H_list))))
    return fs_dict

if __name__ == "__main__":
    TEETH_NUMBER_LIST = [29,95,36,90]
    SPEED = 40
    fs_list = get_fs(TEETH_NUMBER_LIST, SPEED)
    print(fs_list)
    fs_dict = get_split_coordinate(fs_list, 1300, 209)
    print(fs_dict)

    TEETH_NUMBER_LIST = [29,95,36,90]
    SPEED = 40
    FS_MAX = 1300
    label_list = pd.DataFrame
    data= SWJTU_Gear_Data(filename='test_data.csv',label_file_name='test_data_label.csv',file_path='../data/')
    for data_id in ['1','2','3','4']:
        l_t, l_f = data.plt_stft(data_id, '../result', FS_MAX)



