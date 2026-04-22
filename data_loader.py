
import os
import random
import numpy as np
import librosa
import tensorflow as tf
from config import SAMPLE_RATE, SAMPLES, N_MELS, N_FFT, HOP_LENGTH
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

AUTOTUNE = tf.data.experimental.AUTOTUNE

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def load_audio_file(path, sr=SAMPLE_RATE, duration=None):
    audio, _ = librosa.load(path, sr=sr)
    if duration is not None:
        target_len = int(sr * duration)
        if len(audio) < target_len:
            pad_width = target_len - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            audio = audio[:target_len]
    return audio

def wav_to_log_mel(audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)


def extract_features_from_file(path, duration=SAMPLES / SAMPLE_RATE, use_augmentation=False):
    audio = load_audio_file(path, duration=duration)
    if use_augmentation:
        audio = augment(samples=audio, sample_rate=SAMPLE_RATE)
    log_mel = wav_to_log_mel(audio, sr=SAMPLE_RATE)
    return np.expand_dims(log_mel, -1)

def list_audio_files_and_labels(base_dir):
 
    classes = ["car", "not_car"]
    filepaths = []
    labels = []

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    return filepaths, labels, classes

def generator_from_dir(base_dir, duration=SAMPLES/ SAMPLE_RATE, shuffle=True, augment=False):
    filepaths, labels, classes = list_audio_files_and_labels(base_dir)
    data = list(zip(filepaths, labels))
    if shuffle:
        random.shuffle(data)
    for fp, lbl in data:
        feat = extract_features_from_file(fp, duration=duration, use_augmentation=augment)
        yield feat, lbl


def build_tf_dataset(base_dir, batch_size=32, shuffle=True, augment=False):
    filepaths, labels, classes = list_audio_files_and_labels(base_dir)
    if len(filepaths) == 0:
        raise RuntimeError(f"No audio files found in {base_dir}.")
    
    sample_feat = extract_features_from_file(filepaths[0])
    feat_shape = sample_feat.shape

    def _gen():
        for feat, lbl in generator_from_dir(base_dir, shuffle=shuffle, augment=augment):
            yield feat, lbl

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=(feat_shape, ())
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classes