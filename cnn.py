import os
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks # type: ignore
from sklearn.model_selection import train_test_split

PATH = './data'
EPOCHS = 20
LEARNING_RATE = 0.001

def load():
    x = np.load(os.path.join(PATH, 'features.npy'))
    y = np.load(os.path.join(PATH, 'labels.npy'))
    g = np.load(os.path.join(PATH, 'genders.npy'))
    a = np.load(os.path.join(PATH, 'actors.npy'))
    return x, y, g, a

def split(x, y, a, g):
    x_train, x_temp, y_train, y_temp, a_train, a_temp, g_train, g_temp = train_test_split(x, y, a, g, test_size=0.2, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test, a_val, a_test, g_val, g_test = train_test_split(x_temp, y_temp, a_temp, g_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val)

def init(shape):
    model = models.Sequential()

    # 1. Normalization (The Input Scaler)
    model.add(layers.Normalization(axis=None, input_shape=shape))

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

    # 4. Classifier (The Output)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax')) # 8 Emotions

    # Compile the model
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    x, y, a, g = load()
    (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val) = split(x, y, a, g)
    model = init(x.shape[1:])
    checkpoint = callbacks.ModelCheckpoint('data/weights.keras', save_best_only=True)
    model.layers[0].adapt(x_train)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=2
    )

if __name__ == "__main__":
    main()
