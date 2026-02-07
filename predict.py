import numpy as np
import preprocessing
from tensorflow.keras import models # type: ignore

EMOTIONS = ('Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised')
PATH = './data/weights.keras'

def get_data():
    path = input('Enter path to audio file: ')
    path = path.strip()
    data = preprocessing.load(path)
    data = preprocessing.extract(data)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)
    return data

def predict(data):
    model = models.load_model(PATH)
    predictions = model.predict(data, verbose=0)
    return predictions

def output(predictions):
    idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    emotion = EMOTIONS[int(idx)]
    print("-" * 30)
    print(f"Prediction: {emotion}")
    print(f"Confidence: {confidence}%")
    print("-" * 30)

def main():
    data = get_data()
    predictions = predict(data)
    output(predictions)

if __name__ == "__main__":
    main()
