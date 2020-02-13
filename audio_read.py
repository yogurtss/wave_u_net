import soundfile as sf
import os
import cv2
import musdb
import matplotlib.pyplot as plt
import numpy as np
import wave
import librosa


def mu_law_encode(audio, quantization_channels=256):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1)*1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return signal

name_list = []
data_wav = 'musdb18wav/train/'
for root, dirs, files in os.walk(data_wav):
    for dir in dirs:
        name_list.append(dir)
print(data_wav.format('vocals'))
sf_full, _ = librosa.load(data_wav+name_list[0]+'/linear_mixture.wav')
sf_acc, _ = librosa.load(data_wav+name_list[0]+'/accompaniment.wav')
sf_voc, _ = librosa.load(data_wav+name_list[0]+'/vocals.wav')
# sf_full = sf_full * 1.0 / np.max(np.abs(sf_full), axis=0)
sf_full = sf_full[0:16384 * 20]
sf_voc = sf_voc[0:16384 * 20]
sf_acc = sf_acc[0:16384 * 20]
sub_0 = sf_full - sf_acc - sf_voc
sf_full_mono = np.mean(sf_full[0:16384 * 20], 1)
# sf_full_mono = sf_full_mono / np.max(np.abs(sf_full_mono), axis=0)
# sf_voc_mono = np.mean(sf_voc[0:1000], 1)
# sf_acc_mono = np.mean(sf_acc[0:1000], 1)
# sf_full_mono = (sf_full[:, 0] + sf_full[:, 1])/2
# sf_acc_mono = (sf_acc[:, 0] + sf_acc[:, 1])/2
# sf_voc_mono = (sf
# _voc[:, 0] + sf_acc[:, 1])/2
# sub = sf_full_mono - sf_acc_mono - sf_voc_mono
time = np.arange(0, 16384 * 20)
plt.plot(time, sf_full_mono, c='b')
plt.show()
sf_mu = mu_law_encode(sf_full_mono)
plt.plot(time, sf_mu, c='b')
plt.show()
print('test')
