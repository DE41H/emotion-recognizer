# Speech Emotion Recognition (SER) 🎙️

This repository contains a deep learning pipeline for Speech Emotion Recognition (SER). The project uses a 2D Convolutional Neural Network (CNN) to classify human emotions from audio recordings. The model is trained on the RAVDESS dataset and evaluates 8 distinct emotional states.

## 🧠 Project Overview

The system processes raw `.wav` audio files, extracts Mel-spectrograms along with their delta features, and feeds them into a custom-built 2D CNN. It includes an extensive data augmentation pipeline to ensure the model generalizes well and includes a specific evaluation for pitch/gender bias to ensure fairness in predictions.

**Supported Emotions:**
Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

### Key Features
* **Comprehensive Data Augmentation**: Includes white noise injection, pitch shifting, time stretching/shrinking, audio shifting, volume gain changes, audio cutouts, and spectrogram augmentation (frequency and time masking).
* **Robust CNN Architecture**: Built using TensorFlow/Keras with `BatchNormalization`, `SpatialDropout2D`, and L2 regularization to prevent overfitting.
* **Bias Analysis**: Contains built-in testing features to compare the accuracy of predictions on female versus male voices.
* **Real-time Inference Script**: A standalone Python script (`predict.py`) for easy testing on single audio files.

---

## 📂 Repository Structure

* `task.ipynb`: The core Jupyter Notebook containing the entire machine learning pipeline. This handles data loading, augmentation, model definition, training, and evaluation (classification reports, confusion matrix, and bias analysis).
* `predict.py`: A command-line script for testing the trained model on custom, single `.wav` files. *(Note: Requires the custom `preprocessing.py` module containing the `load` and `extract` functions from the notebook).*
* `data/`: Directory where the preprocessed datasets (`dataset.npz`) and the saved model weights (`weights.keras`) are stored.
* `ravdess/`: The expected directory for the raw RAVDESS dataset audio files.

---

## ⚙️ Prerequisites

Make sure you have Python 3.x installed. You will need the following libraries to run this project:

```bash
pip install numpy librosa matplotlib tensorflow scikit-learn
```

## 📊 Dataset Setup

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Download the audio-only files from the RAVDESS dataset.

Extract the downloaded files into a folder named ravdess in the root directory of this project.

The notebook automatically parses the standard RAVDESS file naming convention (e.g., 03-01-05-01-01-01-01.wav) to extract the emotion label, actor ID, and gender.

- Training: Actors 1-18 (Includes data augmentation)

- Testing: Actors 19-21

- Validation: Actors 22+

## 🚀 Usage
1. Training and Testing the Model

Open and run the task.ipynb notebook. The notebook will prompt you with a menu:

```
1 => Train
2 => Test
3 => Train + Test
```

    Option 1/3 (Train): Reads the ravdess folder, processes and augments the audio, saves the arrays to data/dataset.npz, and trains the CNN. It saves the best-performing model to data/weights.keras.

    Option 2/3 (Test): Loads the saved model and evaluates it against the unseen test set. It outputs a classification report, text-based confusion matrix, and a Pitch/Gender bias analysis.

2. Running Inference on Custom Audio

To predict the emotion of a new audio file, use the standalone prediction script:

```
python predict.py
```

You will be prompted to enter the path to your .wav file:

```
Enter path to audio file: path/to/your/audio.wav
------------------------------
Prediction: Happy
Confidence: 94.2%
------------------------------
```

## 🏗️ Model Architecture

The model relies on visual representations of audio (Mel-spectrograms). It expects an input shape dictated by the length of the audio and the number of Mel bands (256).

Input processing: Features are normalized directly inside the model using layers.Normalization.

Feature Extraction: 4 Blocks of Conv2D layers (filters: 32, 64, 128, 256) combined with BatchNormalization, ELU activation, MaxPooling2D, and SpatialDropout2D.

Pooling: Concatenation of both GlobalAveragePooling2D and GlobalMaxPooling2D.

Classification: Fully connected Dense layers (256, 128) with Dropout and L2 regularization, outputting to a final 8-node Softmax layer.

## 📈 Callbacks & Optimization

Optimizer: Adam with an initial learning rate of 0.001.

Dynamic Learning Rate: ReduceLROnPlateau halves the learning rate if validation loss plateaus.

Early Stopping: Halts training if validation loss doesn't improve for 15 epochs, restoring the best weights.

Class Weights: Uses Scikit-learn's compute_class_weight to ensure balanced learning across all emotion classes.