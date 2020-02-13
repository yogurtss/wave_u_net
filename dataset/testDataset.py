import torch
import numpy as np
import torch.nn as nn
import torchvision
import soundfile as sf


class TestDataset(nn.Module):
    def __init__(self, soundlist, root_dir, transform=None):
        super(TestDataset, self).__init__()
        self.sound_list = soundlist
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sound_list)

    def __getitem__(self, idx):
        sound_name = self.sound_list[idx]
        mix_path = self.root_dir + sound_name + '/mix.wav'
        acc_path = self.root_dir + sound_name + '/accompaniment.wav'
        vocal_path = self.root_dir + sound_name + '/vocals.wav'
        mix, _ = sf.read(mix_path, dtype='float32')
        acc, _ = sf.read(acc_path, dtype='float32')
        voc, _ = sf.read(vocal_path, dtype='float32')
        mm = np.random.uniform(0.8, 1) * acc + np.random.uniform(0.8, 1) * voc
        diff = mix - mm
        max_diff = np.max(diff)
        # mix = mix.T
        # acc = acc.T
        # mix = np.mean(mix, 1)
        # acc = np.mean(acc, 1)
        mix = torch.tensor(mix).view(1, 1, -1)
        acc = torch.tensor(acc).view(1, 1, -1)
        mm = torch.tensor(mm).view(1, 1, -1)
        return mix, acc, mm
