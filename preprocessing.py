import os
import numpy as np
import librosa as lb


PATH = './ravdess'
SAMPLE_RATE = 22050
DURATION = 3
LENGTH = DURATION * SAMPLE_RATE
NOISE_FACTOR = 0.005

features = []
labels = []

def load():
    data, _ = lb.load(PATH, sr=SAMPLE_RATE, duration=DURATION)
    data, _ = lb.effects.trim(data)
    if len(data) < LENGTH:
        padding = LENGTH - len(data)
        data = np.pad(data, (0, padding), 'constant')
    return data

def extract(data):
    graph = lb.feature.melspectrogram(y=data, sr=SAMPLE_RATE)
    graph = lb.power_to_db(graph)
    return graph[..., np.newaxis]

def augument(data):
    noise = np.random.randn(LENGTH)
    data_noise = np.array(data + NOISE_FACTOR * noise)
    data_pitch = lb.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=-2)
    return (data_noise, data_pitch)

def parse(graph):
    pass

def save():
    x = np.array(features)
    y = np.array(labels)
    np.save('data/features.npy', x)
    np.save('data/labels.npy', y)

def main():
    data = load()
    data = augument(data)
    graph = extract(data)
    parse(graph)
    save()

if __name__ == "__main__":
    main()
