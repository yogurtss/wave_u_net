import torch
import numpy as np
import torch.nn as nn
import torchvision
import soundfile as sf
import torch.nn.functional as F
import librosa

DownSample = 22050
SampleSize = 16384 * 60  # Around 89s 16384 * 120


class AudioDataset(nn.Module):
    def __init__(self, soundlist, root_dir, transform=None):
        super(AudioDataset, self).__init__()
        self.sound_list = soundlist
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sound_list)

    def __getitem__(self, idx):
        sound_name = self.sound_list[idx]
        vocals_path = self.root_dir + sound_name + '/vocals.wav'
        accompaniment_path = self.root_dir + sound_name + '/accompaniment.wav'
        mix_path = self.root_dir + sound_name + '/mix.wav'
        vocals, _ = sf.read(vocals_path, dtype='float32')
        mixture, _ = sf.read(mix_path, dtype='float32')
        accompaniment, _ = sf.read(accompaniment_path, dtype='float32')
        # m,n = np.max(mixture), np.min(mixture)
        # vocals = np.mean(vocals, 1)
        # accompaniment = np.mean(accompaniment, 1)
        # factor0 = np.random.uniform(0.8, 1)
        # factor1 = np.random.uniform(0.8, 1)
        # mixture = vocals * factor0 + accompaniment * factor1
        # mixture = np.clip(mixture, -1.0, 1.0)
        # accompaniment = np.clip(accompaniment, -1.0, 1.0)
        # mixture = mixture

        if mixture.shape[0] <= SampleSize:
            # Extend the size to match the requirement
            while mixture.shape[0] <= SampleSize:
                mixture = np.concatenate((mixture, mixture))
                # vocals = np.concatenate((vocals, vocals))
                accompaniment = np.concatenate((accompaniment, accompaniment))
        start = np.random.randint(0, mixture.shape[0] - SampleSize + 1)
        mixture = mixture[start:start+SampleSize]
        accompaniment = accompaniment[start:start+SampleSize]
        # mixture = mixture.T
        # accompaniment = accompaniment.T
        mixture = torch.tensor(mixture).view(1, -1)
        # vocals = torch.tensor(vocals).view(1, -1)
        accompaniment = torch.tensor(accompaniment).view(1, -1)
        return mixture, accompaniment


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    MODEL_PATH = '../result/model.pth'
    MODEL_MIN_PATH = '../result/model_min.pth'
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
    train_save = '../musdb18wav/train_down'
    test_save = '../musdb18wav/test_down'
    for idx in range(len(test_list)):
        print(idx)
        sound_name = test_list[idx]
        vocals_path = TEST_PATH + sound_name + '/vocals.wav'
        accompaniment_path = TEST_PATH + sound_name + '/accompaniment.wav'
        vocals, _ = librosa.load(vocals_path, sr=DownSample, dtype=np.float32)
        accompaniment, _ = librosa.load(accompaniment_path, sr=DownSample, dtype=np.float32)

        sound_path = test_save + '/' + sound_name
        if not os.path.exists(sound_path):
            os.mkdir(sound_path)
        vocal_save = sound_path + '/vocals.wav'
        accompaniment_save = sound_path + '/accompaniment.wav'

        librosa.output.write_wav(vocal_save, vocals, sr=DownSample)
        librosa.output.write_wav(accompaniment_save, accompaniment, sr=DownSample)

    '''
    dataset = AudioDataset(train_list, TRAIN_PATH)
    mix, vo, raw = dataset[0]
    time = np.arange(0, 1000)
    plt.plot(time, mix, c='b')
    plt.show()
    plt.plot(time, raw, c='b')
    plt.show()
    '''



