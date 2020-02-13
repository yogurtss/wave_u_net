import torch
import numpy as np
import torch.nn as nn
import torchvision
import soundfile as sf
import torch.nn.functional as F
import librosa
from tqdm import tqdm

DownSample = 22050
SampleSize = 16384


class AudioNewDataset(nn.Module):
    def __init__(self, sound_list, root_dir, transform=None):
        super(AudioNewDataset, self).__init__()
        self.sound_list = sound_list
        self.root_dir = root_dir
        self.transform = transform
        print("Preparing dataset...")
        mix_arr = np.zeros([1, SampleSize], dtype=np.float32)
        acc_arr = np.zeros([1, SampleSize], dtype=np.float32)
        for i in tqdm(range(len(sound_list))):
            sound_name = sound_list[i]
            accompaniment_path = self.root_dir + sound_name + '/accompaniment.wav'
            mix_path = self.root_dir + sound_name + '/mix.wav'
            mixture, _ = sf.read(mix_path, dtype='float32')
            accompaniment, _ = sf.read(accompaniment_path, dtype='float32')

            if mixture.shape[0] % SampleSize != 0:
                pad_frame = SampleSize - mixture.shape[0] % SampleSize
                mixture = np.pad(mixture, (pad_frame // 2, pad_frame - pad_frame // 2))
                accompaniment = np.pad(accompaniment, (pad_frame // 2, pad_frame - pad_frame // 2))
            mixture = np.reshape(mixture, [int(mixture.shape[0] / SampleSize), SampleSize])
            accompaniment = np.reshape(accompaniment, [int(accompaniment.shape[0] / SampleSize), SampleSize])
            mix_arr = np.concatenate([mix_arr, mixture], axis=0)
            acc_arr = np.concatenate([acc_arr, accompaniment], axis=0)
        self.mix_arr = mix_arr[1:]
        self.acc_arr = acc_arr[1:]

    def __len__(self):
        return self.mix_arr.shape[0]

    def __getitem__(self, idx):
        mix = self.mix_arr[idx]
        acc = self.acc_arr[idx]
        mix = torch.tensor(mix).view(1, -1)
        acc = torch.tensor(acc).view(1, -1)
        return mix, acc


if __name__ == '__main__':
    import os

    TRAIN_PATH = '../musdb18wav/train/'
    TEST_PATH = '../musdb18wav/test/'
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(TRAIN_PATH):
        for dir in dirs:
            train_list.append(dir)

    for root, dirs, files in os.walk(TEST_PATH):
        for dir in dirs:
            test_list.append(dir)
    dataset = AudioNewDataset(train_list, TRAIN_PATH)
    a, b = dataset[0]
