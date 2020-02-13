import librosa
import numpy as np
import soundfile


def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")


def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr


def crop(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["start_frame"]:shapes["end_frame"]]
    return mix, targets


def random_amplify(mix, targets, min, max, shapes):
    '''
    Data augmentation by randomly amplifying sources before adding them to form a new mixture
    :param mix: Original mixture (optional, will be discarded)
    :param targets: Source targets
    :param shapes: Shape dict from model
    :param min: Minimum possible amplification
    :param max: Maximum possible amplification
    :return: New data point as tuple (mix, targets)
    '''
    new_mix = 0
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key] * np.random.uniform(min, max)
            new_mix += targets[key]
    return crop(new_mix, targets, shapes)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step
