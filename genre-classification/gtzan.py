import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Reshape
from keras.utils import to_categorical

class GenreClassifier:
    def __init__(self, gtzan_data_path, local_data_path):
        self.gtzan_data_path = gtzan_data_path
        self.local_data_path = local_data_path

    def load_gtzan_data(self):
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        X, y = [], []
        for genre in genres:
            genre_path = os.path.join(self.gtzan_data_path, genre)
            for filename in os.listdir(genre_path):
                file_path = os.path.join(genre_path, filename)
                audio, _ = librosa.load(file_path, sr=22050, duration=30)  # Load audio (30 seconds)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
                # Pad or truncate mel-spectrogram to a fixed length (e.g., 1292 frames)
                mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=1292, axis=1)
                X.append(mel_spectrogram)
                y.append(genre)
        return np.array(X), np.array(y)

    def preprocess_data(self):
        X, y = self.load_gtzan_data()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(128, 1292, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Reshape((128, -1)))  # Reshape layer
        model.add(LSTM(64, return_sequences=True))  # First LSTM layer
        model.add(LSTM(32))  # Second LSTM layer
        model.add(Dense(10, activation="softmax"))  # 10 genres
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        X_train = X_train.reshape(X_train.shape[0], 128, 1292, 1)
        X_test = X_test.reshape(X_test.shape[0], 128, 1292, 1)
        y_train_cat = to_categorical(y_train, num_classes=10)
        y_test_cat = to_categorical(y_test, num_classes=10)

        model = self.build_model()
        model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=200, batch_size=32)
        return model

if __name__ == "__main__":
    gtzan_data_path = "/data/gtzan"  # Update with actual path
    local_data_path = "/data"  # Update with actual path

    genre_classifier = GenreClassifier(gtzan_data_path, local_data_path)
    trained_model = genre_classifier.train_model()
    print("Genre classification model trained successfully!")
