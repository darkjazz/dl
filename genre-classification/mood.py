import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Reshape
from keras.utils import to_categorical
import psycopg2

class GenreClassifier:
    def __init__(self, mood_audio_path, max_size, n_mels=128):
        self.mood_audio_path = mood_audio_path
        self.max_size = max_size
        self.n_mels = n_mels
        self.pg = Psql()
        self.le = LabelEncoder()
        self.sr = 22050
        self.batch_size = 32
        self.X, self.y = [], []

    def load_mood_data(self):
        for filename in os.listdir(self.mood_audio_path):
            print(filename)
            tags = self.get_tags(filename)
            if tags:
                file_path = os.path.join(self.mood_audio_path, filename)
                audio, _ = librosa.load(file_path, sr=self.sr, duration=30)  # Load audio (30 seconds)
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sr,
                    n_mels=self.n_mels
                )
                # Pad or truncate mel-spectrogram to a fixed length (e.g., 1292 frames)
                mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=self.max_size, axis=1)
                self.X.append(mel_spectrogram)
                self.y.append(tags[0])
        return np.array(self.X), np.array(self.y)

    def get_tags(self, name):
        results = self.pg.execute_query(
            f"select tags from mood.mbtags where name = '{name}'"
        )
        if results:
            results = results[0]
        return results
    def preprocess_data(self):
        X, y = self.load_mood_data()
        y_encoded = self.le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(self.n_mels, self.max_size, 1)))
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
        model.add(Dense(self.le.classes_.shape[0], activation="softmax"))  # 10 genres
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
    def train_model(self, epochs):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        X_train = X_train.reshape(X_train.shape[0], self.n_mels, self.max_size, 1)
        X_test = X_test.reshape(X_test.shape[0], self.n_mels, self.max_size, 1)
        y_train_cat = to_categorical(y_train, num_classes=self.le.classes_.shape[0])
        y_test_cat = to_categorical(y_test, num_classes=self.le.classes_.shape[0])

        model = self.build_model()
        model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=self.batch_size
        )
        return model


class Psql:
    def connect_pg(self):
        return psycopg2.connect(
            host="localhost",
            dbname="synth",
            user="postgres",
            password="p05tgr35",
            application_name="genre-classifier"
        )

    def execute_query(self, query):
        results = []
        try:
            conn = self.connect_pg()
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
        except Exception as e:
            print(e)
        finally:
            if conn:
                conn.close()
        return [ _r[0] for _r in results ]


class TagAnalyser:
    def __init__(self):
        self.pg = Psql()

    def get_all_tags(self):
        self.tags = {}
        names = self.pg.execute_query(
            "select name from mood.mbtags"
        )
        for name in names:
            tags = self.get_tags(name)
            if tags:
                self.tags[name] = tags

    def get_tags(self, name):
        results = self.pg.execute_query(
            f"select tags from mood.mbtags where name = '{name}'"
        )
        if results:
            results = results[0]
        return results

if __name__ == "__main__":
    audio_path = "/audio"  # Update with actual path

    genre_classifier = GenreClassifier(audio_path, 1292)
    trained_model = genre_classifier.train_model(200)
    print("Genre classification model trained successfully!")
