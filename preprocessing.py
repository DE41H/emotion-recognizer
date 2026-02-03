import os
import numpy as np
import librosa as lb

PATH = './ravdess'
SAMPLE_RATE = 22050
DURATION = 3
LENGTH = DURATION * SAMPLE_RATE
NOISE_FACTOR = 0.005
STRETCH_FACTOR = 1.2
SHRINK_FACTOR = 0.8
SHIFT_FACTOR = 0.2
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
    data, _ = lb.load(path, sr=SAMPLE_RATE, duration=None)
    data, _ = lb.effects.trim(data)
    data = fix(data)
    return data

def fix(data):
    if len(data) < LENGTH:
        padding = LENGTH - len(data)
        data = np.pad(data, (0, padding), 'constant')
    elif len(data) > LENGTH:
        data = data[:LENGTH]
    return data

def extract(data):
    graph = lb.feature.melspectrogram(y=data, sr=SAMPLE_RATE, n_mels=128)
    graph = lb.power_to_db(graph)
    return graph[..., np.newaxis]

def augument(data):
    full_data = []
    full_data.append(data)
    noise = np.random.randn(LENGTH)
    data_noise = np.array(data + NOISE_FACTOR * noise)
    full_data.append(data_noise)
    data_deep = lb.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=-2)
    full_data.append(data_deep)
    data_shrill = lb.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=+2)
    full_data.append(data_shrill)
    data_fast = lb.effects.time_stretch(data, rate=STRETCH_FACTOR)
    data_fast = fix(data_fast)
    full_data.append(data_fast)
    data_slow = lb.effects.time_stretch(data, rate=SHRINK_FACTOR)
    data_slow = fix(data_slow)
    full_data.append(data_slow)
    shift = np.random.randint(int(SAMPLE_RATE * SHIFT_FACTOR))
    data_shift = np.roll(data, shift)
    if shift > 0:
        data_shift[:shift] = 0
    data_shift = fix(data_shift)
    full_data.append(data_shift)
    data_inverted = data * -1
    full_data.append(data_inverted)
    gain_factor = np.random.uniform(0.8, 1.2)
    data_gain = data * gain_factor
    full_data.append(data_gain)
    return full_data

def parse(root, filename):
    if not filename.endswith('.wav'):
        return
    args = filename.removesuffix('.wav').split('-')
    emotion = args[2]
    actor = args[6]
    gender = 0 if int(actor) % 2 == 0 else 1
    if emotion not in EMOTIONS:
        return
    path = os.path.join(root, filename)
    data = load(path)
    full_data = augument(data)
    for item in full_data:
        features.append(extract(item))
        labels.append(EMOTIONS[emotion])
        actors.append(actor)
        genders.append(gender)
        
def save():
    x = np.array(features)
    y = np.array(labels)
    z = np.array(genders)
    a = np.array(actors)
    os.makedirs('data', exist_ok=True)
    np.save('data/features.npy', x)
    np.save('data/labels.npy', y)
    np.save('data/genders.npy', z)
    np.save('data/actors.npy', a)

def main():
    progress = 0
    for root, _, files in os.walk(PATH):
        for file in files:
            parse(root, file)
            progress += 1
        print(f'{(progress * 100) / (24 * 60)}% Complete [{24 * 60 - progress} files remaining]')
    save()

if __name__ == "__main__":
    main()
