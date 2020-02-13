import time
from tqdm import tqdm
import soundfile as sf
import os
import numpy as np
import librosa
from torch.utils.data import DataLoader
from dataset.audioDatasetHdf import AudioDataset
import torch
from models.wave_net import WaveUNet
from config import cfg
from dataset.util import *

MODEL_PATH = 'result/2020-02-10-01/model_best.pth'
args = cfg()


def validate(song_path, args):
    shapes = {'start_frame': 6140,
              'end_frame': 51201,
              'output_len': 45061,
              'input_len': 57341}

    INSTRUMENTS = {"bass": False,
                   "drums": False,
                   "other": False,
                   "vocals": True,
                   "accompaniment": True}
    # Create dataloader

    # Read song

    mix, _ = librosa.load(song_path, sr=args.sr, mono=True)
    length = mix.shape[0]

    # Create Model and Load parameters
    print("Creating and Loading model")
    state_dict = torch.load(MODEL_PATH)
    state_pa = state_dict['model_state_dict']
    model = WaveUNet(training=False)
    model.load_state_dict(state_pa)
    # model = model.cuda()

    if length % shapes['output_len'] != 0:
        pad_back = shapes['output_len'] - length % shapes['output_len']
    else:
        pad_back = 0

    mix = np.concatenate([mix, np.zeros(pad_back, dtype=np.float32)])
    output_frames = mix.shape[0]
    # n = mix.shape[0] // shapes['input_len']
    pad_front_suit = shapes['start_frame']
    pad_back_suit = shapes['input_len'] - shapes['end_frame']
    mix = np.concatenate([np.zeros(pad_front_suit, dtype=np.float32), mix, np.zeros(pad_back_suit, dtype=np.float32)], axis=0)

    mix = torch.tensor(mix).view(1, 1, -1)
    vocals = np.zeros(output_frames, dtype=np.float32)
    accompaniment = np.zeros(output_frames, dtype=np.float32)

    for start_frame in tqdm(range(0, output_frames, shapes['output_len'])):
        end_frame = start_frame + shapes['input_len']
        x = mix[..., start_frame:end_frame]
        output = model(x)
        # voc = output[0, 0, :].detach().numpy()
        acc = output[0, 0, :].detach().numpy()
        # vocals[start_frame: start_frame+shapes['output_len']] = voc
        accompaniment[start_frame: start_frame+shapes['output_len']] = acc
    # soundfile.write('result/vocals.wav', vocals, samplerate=args.sr)
    soundfile.write('result/accompaniment.wav', accompaniment, samplerate=args.sr)
    print('Finish...')


if __name__ == '__main__':
    song_path = 'musdb18hq/test/Carlos Gonzalez - A Place For Us/mixture.wav'
    # song_path = 'result/mix.wav'
    validate(song_path, args)
