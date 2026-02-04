import os
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks # type: ignore
from sklearn.model_selection import StratifiedGroupKFold

var = int(input("1 => Train\n2 => Test\n\n: "))

PATH = './data'
EPOCHS = 50
LEARNING_RATE = 0.001

def load():
    x = np.load(os.path.join(PATH, 'features.npy'))
    y = np.load(os.path.join(PATH, 'labels.npy'))
    g = np.load(os.path.join(PATH, 'genders.npy'))
    a = np.load(os.path.join(PATH, 'actors.npy'))
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)
    return x, y, a, g

def split(x, y, a, g):
    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    train_temp_idx, test_idx = next(sgkf_outer.split(x, y, groups=a))
    x_test = x[test_idx]
    y_test = y[test_idx]
    a_test = a[test_idx]
    g_test = g[test_idx]
    x_temp = x[train_temp_idx]
    y_temp = y[train_temp_idx]
    a_temp = a[train_temp_idx]
    g_temp = g[train_temp_idx]
    sgkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf_inner.split(x_temp, y_temp, groups=a_temp))
    x_train = x_temp[train_idx]
    y_train = y_temp[train_idx]
    a_train = a_temp[train_idx]
    g_train = g_temp[train_idx]
    x_val = x_temp[val_idx]
    y_val = y_temp[val_idx]
    a_val = a_temp[val_idx]
    g_val = g_temp[val_idx]
    return (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val)

def init(shape):
    model = models.Sequential()

    # 1. Normalization (The Input Scaler)
    model.add(layers.Normalization(axis=None, input_shape=shape))

    # 2. Convolutional Blocks
    model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # 2. Convolutional Blocks (The Feature Extractors)
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # 3. Global Average Pooling (The Efficient Summarizer)
    # This averages the features instead of flattening them, saving memory.
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    # 4. Classifier (The Output)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax')) # 8 Emotions

    # Compile the model
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train(model, x_train, x_val, y_train, y_val):
    checkpoint = callbacks.ModelCheckpoint('data/weights.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    dynamic_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=64,
        callbacks=[checkpoint, dynamic_lr, early_stop],
        verbose=1
    )
    return history

def test(model, x_test, y_test):
    print(f'Testing Model....')
    model.load_weights('data/weights.keras')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Final Accuracy: {test_acc}\tFinal Loss: {test_loss}')

def main():
    x, y, a, g = load()
    (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val) = split(x, y, a, g)
    model = init(x.shape[1:])
    model.layers[0].adapt(x_train)
    if var == 1:
        train(model, x_train, x_val, y_train, y_val)
    elif var == 2:
        test(model, x_test, y_test)

if __name__ == "__main__":
    main()
