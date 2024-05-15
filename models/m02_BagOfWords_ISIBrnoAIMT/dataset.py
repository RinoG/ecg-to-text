import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import numpy as np
from scipy.stats import zscore

class PtbXlDataset(Dataset):
    def __init__(self, dataset_path, mode='train', n_BoW=50):
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode must be one of 'train', 'val', 'test'")

        self.dataset_path = dataset_path
        self.mode = mode
        self.n_BoW = n_BoW

        self.df_data = pd.read_csv(f'{dataset_path}/{mode}.csv')
        self.df_BoW = pd.read_csv(f'{dataset_path}/bag_of_words/{mode}_{n_BoW}_BoW.csv')

        mismatches = self.df_data['ecg_id'] != self.df_BoW['ecg_id']
        if mismatches.any():
            mismatched_indices = mismatches[mismatches].index.tolist()
            raise ValueError(f"ECG ID mismatch at indices: {mismatched_indices}")

        # # Butterworth Filter Init
        # ecg_signal_path = self.df_data.iloc[0]["filename_hr"]
        # fs = wfdb.rdrecord(self.dataset_path + ecg_signal_path).fs
        # low_cutoff = 1
        # high_cutoff = 47
        # nyquist_freq = fs / 2
        # w_n = (low_cutoff / nyquist_freq, high_cutoff / nyquist_freq)
        # self.butter_b, self.butter_a = signal.butter(N=3, Wn=w_n, btype='bandpass')

    def __len__(self):
        return len(self.df_BoW)

    def __getitem__(self, idx):
        BoW = self.df_BoW.iloc[idx].values[1:]

        ecg_signal_path = self.df_data.iloc[idx]["filename_hr"]
        ecg_signal = wfdb.rdrecord(self.dataset_path + ecg_signal_path).p_signal
        # ecg_signal = self.preprocessing(ecg_signal.transpose())

        return ecg_signal.transpose(), BoW

    def summary(self, output):
        if output == 'pandas':
            return pd.Series(self.df_BoW.drop('ecg_id', axis=1).to_numpy().sum(axis=0),index=self.df_BoW.columns[1:])
        if output == 'numpy':
            return self.df_BoW.drop('ecg_id', axis=1).to_numpy().sum(axis=0)

    # def preprocessing(self, recording, fs=500):
    #     # if fs == 1000:
    #     #     recording = signal.resample_poly(recording, up=1, down=2, axis=-1)  # to 500Hz
    #     #     fs = 500
    #     # elif fs == 500:
    #     #     pass
    #     # else:
    #     #     recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
    #
    #     recording = signal.filtfilt(self.butter_b, self.butter_a, recording)
    #     # recording = zscore(recording, axis=-1)
    #     recording = np.nan_to_num(recording)
    #     # recording = zero_padding(recording)
    #
    #     return recording


if __name__ == "__main__":
    ptbxl_dataset = PtbXlDataset('ptbxl_dataset.csv', 'train', n_BoW=10)

    dataloader = DataLoader(ptbxl_dataset, batch_size=128, shuffle=True)


