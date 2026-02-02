import os
import numpy as np
import librosa as lb


PATH = './ravdess'
SAMPLE_RATE = 22050
DURATION = 3
LENGTH = DURATION * SAMPLE_RATE
NOISE_FACTOR = 0.005
EMOTIONS = {
    '01': 0,
    '02': 1,
    '03': 2,
    '04': 3,
    '05': 4,
    '06': 5,
    '07': 6,
    '08': 7
}

features = []
labels = []
genders = []
actors = []

def load(path):
    data, _ = lb.load(path, sr=SAMPLE_RATE, duration=DURATION)
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

def parse(root, filename):
    if not filename.endswith('.wav'):
        return
    args = filename.split('-')
    emotion = args[2]
    actor = args[6]
    gender = 0 if int(actor) % 2 == 0 else 1
    if emotion not in EMOTIONS:
        return
    path = os.path.join(root, filename)
    data = load(path)
    data_noise, data_pitch = augument(data)
    features.append(extract(data))
    features.append(extract(data_noise))
    features.append(extract(data_pitch))
    for _ in range(3):
        labels.append(EMOTIONS[emotion])
        actors.append(actor)
        genders.append(gender)
        
def save():
    x = np.array(features)
    y = np.array(labels)
    z = np.array(genders)
    a = np.array(actors)
    np.save('data/features.npy', x)
    np.save('data/labels.npy', y)
    np.save('data/genders.npy', z)
    np.save('data/actors.npz', a)

def main():
    for root, _, files in os.walk(PATH):
        for file in files:
            parse(root, file)
    save()

if __name__ == "__main__":
    main()
