import os
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks # type: ignore
from sklearn.model_selection import GroupShuffleSplit

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
    return x, y, g, a

def split(x, y, a, g):
    actors = np.unique(a)
    np.random.seed(42)
    np.random.shuffle(actors)
    test_actors = actors[:2]
    val_actors = actors[2:4]
    train_actors = actors[4:]
    test_mask = np.isin(a, test_actors)
    val_mask = np.isin(a, val_actors)
    train_mask = np.isin(a, train_actors)
    x_test, y_test, a_test, g_test = x[test_mask], y[test_mask], a[test_mask], g[test_mask]
    x_val, y_val, a_val, g_val = x[val_mask], y[val_mask], a[val_mask], g[val_mask]
    x_train, y_train, a_train, g_train = x[train_mask], y[train_mask], a[train_mask], g[train_mask]
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
