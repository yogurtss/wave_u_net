import soundfile as sf
import os
import numpy as np

TRAIN_PATH = 'musdb18wav/train_down/'
TEST_PATH = 'musdb18wav/test_down/'

mean_vocal = 0.0
var_vocal = 0.0
mean_acc = 0.0
var_acc = 0.0
mean_mix = 0.0
var_mix = 0.0

train_list = []
for root, dirs, files in os.walk(TRAIN_PATH):
    for dir in dirs:
        train_list.append(dir)

for idx in range(len(train_list)):
    print(idx)
    sound_name = train_list[idx]
    vocals_path = TRAIN_PATH + sound_name + '/vocals.wav'
    accompaniment_path = TRAIN_PATH + sound_name + '/accompaniment.wav'
    vocals, s_v = sf.read(vocals_path)
    acc, s_a = sf.read(accompaniment_path)
    mixture = vocals + acc
    mixture = mixture / np.max(np.abs(mixture))

    try:
        s_v == 22050 and s_a == 22050
    except:
        raise KeyError




