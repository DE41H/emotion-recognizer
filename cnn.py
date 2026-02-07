import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, callbacks # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

var = int(input("1 => Train\n2 => Test\n3 => Train + Test\n\n: "))

PATH = './data'
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EMOTIONS = ('Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised')

def load():
    ds = np.load(os.path.join(PATH, 'dataset.npz'))
    x_train, x_test, x_val = ds['x_train'], ds['x_test'], ds['x_val']
    y_train, y_test, y_val = ds['y_train'], ds['y_test'], ds['y_val']
    a_train, a_test, a_val = ds['a_train'], ds['a_test'], ds['a_val']
    g_train, g_test, g_val = ds['g_train'], ds['g_test'], ds['g_val']
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
    if x_test.ndim == 3:
        x_test = np.expand_dims(x_test, axis=-1)
    if x_val.ndim == 3:
        x_val = np.expand_dims(x_val, axis=-1)
    return (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val)

def init(shape):
    model = models.Sequential()
    model.add(layers.Input(shape=shape))
    model.add(layers.Normalization(axis=None))

    model.add(layers.Conv2D(32, (5, 5), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.3))

    model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.5))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.7))

    model.add(layers.Dense(128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(8, activation='softmax', dtype='float32'))

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train(model, x_train, x_val, y_train, y_val):
    checkpoint = callbacks.ModelCheckpoint('data/weights.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    dynamic_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.000001, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, dynamic_lr, early_stop],
        verbose=1
    )
    return history

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

def test(x_test, y_test, g_test):
    print(f'Testing Model....')
    model = models.load_model('data/weights.keras')
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "-" * 50)
    print("CONFUSION MATRIX (Text)")
    print("-" * 50)
    print(cm)
    female_idx = np.where(g_test == 0)[0]
    male_idx = np.where(g_test == 1)[0]
    female_acc = np.mean(y_pred[female_idx] == y_test[female_idx])
    male_acc = np.mean(y_pred[male_idx] == y_test[male_idx])
    print('\n' + '-' * 50)
    print('PITCH/GENDER BIAS ANALYSIS')
    print('-' * 50)
    print(f'Female Accuracy: {female_acc*100:.2f}%')
    print(f'Male Accuracy: {male_acc*100:.2f}%')

def main():
    (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val) = load()
    if var == 1 or var == 3:
        model = init(x_train.shape[1:])
        model.layers[0].adapt(x_train)
        history = train(model, x_train, x_val, y_train, y_val)
        plot_history(history)
    if var == 2 or var == 3:
        test(x_test, y_test, g_test)

if __name__ == "__main__":
    main()
