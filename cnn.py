import os
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks # type: ignore
from sklearn.model_selection import GroupShuffleSplit

PATH = './data'
EPOCHS = 50
LEARNING_RATE = 0.001

def load():
    x = np.load(os.path.join(PATH, 'features.npy'))
    y = np.load(os.path.join(PATH, 'labels.npy'))
    g = np.load(os.path.join(PATH, 'genders.npy'))
    a = np.load(os.path.join(PATH, 'actors.npy'))
    return x, y, g, a

def split(x, y, a, g):
    gss_1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_id, temp_id = next(gss_1.split(x, y, groups=a))
    x_train, x_temp = x[train_id], x[temp_id]
    y_train, y_temp = y[train_id], y[temp_id]
    a_train, a_temp = a[train_id], a[temp_id]
    g_train, g_temp = g[train_id], g[temp_id]
    gss_2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    test_id, val_id = next(gss_2.split(x_temp, y_temp, groups=a_temp))
    x_test, x_val = x_temp[test_id], x_temp[val_id]
    y_test, y_val = y_temp[test_id], y_temp[val_id]
    a_test, a_val = a_temp[test_id], a_temp[val_id]
    g_test, g_val = g_temp[test_id], g_temp[val_id]
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

def train(model, x_train, x_val, y_train, y_val):
    checkpoint = callbacks.ModelCheckpoint('data/weights.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    dynamic_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    model.layers[0].adapt(x_train)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[checkpoint, dynamic_lr],
        verbose=1
    )

def test(model, x_test, y_test):
    print(f'Testing Model....')
    model.load_weights('data/weights.keras')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Final Accuracy: {test_acc}\tFinal Loss: {test_loss}')

def main():
    x, y, a, g = load()
    (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val) = split(x, y, a, g)
    model = init(x.shape[1:])
    train(model, x_train, x_val, y_train, y_val)
    test(model, x_test, y_test)

if __name__ == "__main__":
    main()
