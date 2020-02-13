import musdb
import soundfile
import os
import librosa as lib
import glob
from tqdm import tqdm
import numpy as np
import h5py
from dataset.util import *
global SAVE_PATH
SAVE_PATH = 'musdb18wav/'
H5_PATH = 'H5/'
INSTRUMENTS = ["bass", "drums", "other", "vocals", "accompaniment"]


def getMUSDBHQ(database_path):
    subsets = list()

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            # Add other instruments to form accompaniment
            acc_path = os.path.join(track_folder, "accompaniment.wav")

            if not os.path.exists(acc_path):
                print("Writing accompaniment to " + track_folder)
                stem_audio = []
                for stem in ["bass", "drums", "other"]:
                    audio, sr = load(example[stem], sr=None, mono=False)
                    stem_audio.append(audio)
                acc_audio = np.clip(sum(stem_audio), -1.0, 1.0)
                write_wav(acc_path, acc_audio, sr)

            example["accompaniment"] = acc_path

            samples.append(example)

        subsets.append(samples)
    train_val_list = subsets[0]
    test_list = subsets[1]

    np.random.seed(42)
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    dataset = {'train': train_list,
               'val': val_list,
               'test': test_list}
    return dataset


def getMUSDB(database_path):
    # 导入数据
    mus = musdb.DB(root=database_path, is_wav=False)

    subsets = list()
    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            # track_path = track.path[:-4]
            track_path = SAVE_PATH + subset + '/' + track.name
            if not os.path.exists(track_path):
                os.mkdir(track_path)
            mix_path = track_path + "/mix.wav"
            acc_path = track_path + "/accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix": mix_path, "accompaniment": acc_path}
                paths.update({key: track_path + "/" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + '/' + stem + ".wav"
                audio = track.targets[stem].audio.T
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio.T
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(
                np.max(diff_signal)))  # Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    train_val_list = subsets[0]
    test_list = subsets[1]

    np.random.seed(42)
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    dataset = {'train': train_list,
               'val': val_list,
               'test': test_list}
    return dataset


def makeH5py(dataset, h5_dir):
    sr = 22050
    channels = 2
    partitions = ['train', 'val', 'test']
    for partition in partitions:
        hdf_dir = os.path.join(h5_dir, partition + ".hdf5")
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)

        with h5py.File(hdf_dir, "w") as f:
            f.attrs["sr"] = sr
            f.attrs["channels"] = channels
            f.attrs["instruments"] = INSTRUMENTS

            for idx, example in enumerate(tqdm(dataset[partition])):
                # Load mix
                mix_audio, _ = load(example["mix"], sr=sr)

                source_audios = []
                for source in INSTRUMENTS:
                    # In this case, read in audio and convert to target sampling rate
                    source_audio, _ = load(example[source], sr=sr)
                    source_audios.append(source_audio)
                source_audios = np.concatenate(source_audios, axis=0)
                assert (source_audios.shape[1] == mix_audio.shape[1])

                # Add to HDF5 file
                grp = f.create_group(str(idx))
                grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                grp.attrs["length"] = mix_audio.shape[1]
                grp.attrs["target_length"] = source_audios.shape[1]
        print('test')


if __name__ == '__main__':
    dataset = getMUSDBHQ('C:\Projects\WaveUNet\musdb18hq')
    makeH5py(dataset, H5_PATH)


