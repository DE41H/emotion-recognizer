import os
import numpy as np
import librosa as lb

PATH = './ravdess'
SAMPLE_RATE = 22050
N_MELS = 128
DURATION = 3
LENGTH = int(DURATION * SAMPLE_RATE)
NOISE_FACTOR = 0.001
STRETCH_FACTOR = 1.2
SHRINK_FACTOR = 0.8
SHIFT_FACTOR = 0.2
CUT_LENGTH = 4000
FREQ_MASK = 25
TIME_MASK = 30
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

x_train, x_test, x_val = [], [], []
y_train, y_test, y_val = [], [], []
g_train, g_test, g_val = [], [], []
a_train, a_test, a_val = [], [], []

def load(path):
    data, _ = lb.load(path, sr=SAMPLE_RATE, duration=None)
    data, _ = lb.effects.trim(data, top_db=30)
    data = fix(data)
    return data

def fix(data):
    if len(data) < LENGTH:
        padding = int(LENGTH - len(data))
        data = np.pad(data, (0, padding), 'constant')
    elif len(data) > LENGTH:
        data = data[:LENGTH]
    return data

def extract(data):
    graph = lb.feature.melspectrogram(y=data, sr=SAMPLE_RATE, n_mels=N_MELS)
    graph = lb.power_to_db(graph)
    delta = lb.feature.delta(graph)
    delta_2 = lb.feature.delta(graph, order=2)
    graph = np.stack([graph, delta, delta_2], axis=-1)
    return graph.astype(np.float32)

def spec_augument(graph):
    spec = graph.copy()
    n_mels = spec.shape[0]
    f = np.random.randint(0, FREQ_MASK)
    f0 = np.random.randint(0, n_mels - f)
    spec[f0:f0 + f, :, :] = 0
    n_time = spec.shape[1]
    t = np.random.randint(0, TIME_MASK)
    t0 = np.random.randint(0, n_time - t)
    spec[:, t0:t0 + t, :] = 0
    return spec

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
    data_cut = data.copy()
    start = np.random.randint(0, LENGTH - CUT_LENGTH)
    stop = start + CUT_LENGTH
    data_cut[start:stop] = 0
    full_data.append(data_cut)
    return full_data

def get_file_data():
    files_data = []
    for root, _, files in os.walk(PATH):
        for file in files:
            if not file.endswith('.wav'):
                continue
            args = file.removesuffix('.wav').split('-')
            emotion = args[2]
            actor = args[6]
            gender = 0 if int(actor) % 2 == 0 else 1
            if emotion not in EMOTIONS:
                continue
            files_data.append({
                'path': os.path.join(root, file),
                'label': EMOTIONS[emotion],
                'actor': int(actor),
                'gender': gender
            })
    return files_data

def parse(file_data, arr_x, arr_y, arr_g, arr_a):
    data = load(file_data['path'])
    graph = extract(data)
    arr_x.append(graph)
    arr_y.append(file_data['label'])
    arr_a.append(file_data['actor'])
    arr_g.append(file_data['gender'])

def parse_and_augument(file_data, arr_x, arr_y, arr_g, arr_a):
    data = load(file_data['path'])
    full_data = augument(data)
    for item in full_data:
        graph = extract(item)
        arr_x.append(graph)
        arr_x.append(spec_augument(graph))
        for _ in range(2):
            arr_y.append(file_data['label'])
            arr_a.append(file_data['actor'])
            arr_g.append(file_data['gender'])
        
def save():
    x = {
        'train': np.array(x_train).astype('float32'),
        'test': np.array(x_test).astype('float32'),
        'val': np.array(x_val).astype('float32'),
    }
    y = {
        'train': np.array(y_train).astype('float32'),
        'test': np.array(y_test).astype('float32'),
        'val': np.array(y_val).astype('float32'),
    }
    a = {
        'train': np.array(a_train).astype('float32'),
        'test': np.array(a_test).astype('float32'),
        'val': np.array(a_val).astype('float32'),
    }
    g = {
        'train': np.array(g_train).astype('float32'),
        'test': np.array(g_test).astype('float32'),
        'val': np.array(g_val).astype('float32'),
    }
    os.makedirs('data', exist_ok=True)
    np.savez_compressed('data/dataset.npz', x_train=x['train'], x_test=x['test'], x_val=x['val'], 
                        y_train=y['train'], y_test=y['test'], y_val=y['val'], 
                        g_train=g['train'], g_test=g['test'], g_val=g['val'], 
                        a_train=a['train'], a_test=a['test'], a_val=a['val'])

def main():
    files_data = get_file_data()
    for file_data in files_data:
        if file_data['actor'] <= 19:
            parse_and_augument(file_data, x_train, y_train, g_train, a_train)
        elif file_data['actor'] <= 22:
            parse(file_data, x_test, y_test, g_test, a_test)
        else:
            parse(file_data, x_val, y_val, g_val, a_val)
    save()

if __name__ == "__main__":
    main()
