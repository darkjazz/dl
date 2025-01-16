import os
import json
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Reshape
from keras.utils import to_categorical


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.exclude = ["pop", "rock", "parody", "unknown", "undefined", "love songs", "promotional", "country comedy", "various", "popular"]

    def load_data(self, tracks_file, tags_file):
        with open(os.path.join(self.data_path, tracks_file)) as f:
            self.tracks = json.load(f)
        with open(os.path.join(self.data_path, tags_file)) as f:
            self.tags = json.load(f)

    def process_tags(self):
        self.collect_tag_stats()

    def collect_tag_stats(self):
        self.tag_stats = { }
        self.tags = { }
        for _tr in self.tracks.values():
            for _tag in _tr.get('tags', []):
                if _tag not in self.tag_stats:
                    self.tag_stats[_tag] = 1
                else:
                    self.tag_stats[_tag] += 1
        self.tags = { _k: _v for _k, _v in sorted(self.tag_stats.items(), key=lambda x:x[1], reverse=True ) }
        self.total_tags = sum([_v for _v in self.tags.values()])

    def which_tag_above_pct(self, percentile):
        total_occ = 0
        total_tags = 0
        for _tag, _occ in self.tags.items():
            total_occ += _occ
            total_tags += 1
            if (total_occ/self.total_tags) > percentile:
                return (_tag, _occ, total_tags)

    def make_cooccurrence_matrix(self):
        n = len(self.tags)
        tag_list = list(self.tags.keys())
        self.cooc_matrix = np.zeros((n, n))
        for track in self.tracks.values():
            tags = track.get('tags', [])
            indices = [tag_list.index(tag) for tag in tags]

            for i, indA in enumerate(indices):
                for indB in indices:
                    if indA != indB:
                        self.cooc_matrix[indA][indB] += 1

    def get_cooccurring_tags(self, tag_name):
        tag_index = list(self.tags.keys()).index(tag_name)
        cooc_row = self.cooc_matrix[tag_index]
        tag_list = list(self.tags.keys())
        cooccurring_tags = {tag_list[i]: int(cooc_row[i]) for i in range(len(cooc_row)) if cooc_row[i] > 0}
        return cooccurring_tags

    def merge_tags(self, tracks_file, merge_file):
        with open(os.path.join(self.data_path, tracks_file)) as f:
            self.tracks = json.load(f)
        with open(os.path.join(self.data_path, merge_file)) as f:
            self.merge_map = json.load(f)
        self.merged_tags = {}
        for track in self.tracks.values():
            tags = [t for t in track.get('tags', []) if t not in self.merge_map and t not in exclude]
            tags.extend([self.merge_map[t] for t in track.get('tags', []) if t in self.merge_map ])
            self.merged_tags[track['arid']] = list(set(tags))
        with open(os.path.join(self.data_path, "merged_tags.json"), "w") as f:
            json.dump(self.merged_tags, f)

    def remerge_tags(self):
        with open(os.path.join(self.data_path, "merged_tags.json")) as f:
            self.source_tags = json.load(f)
        with open(os.path.join(self.data_path, "tag_merge_merge.json")) as f:
            self.merge_map = json.load(f)
        self.merged_tags = {}
        for arid, tags in self.source_tags.items():
            m_tags = [t for t in tags if t not in self.merge_map and t not in self.exclude]
            m_tags.extend([self.merge_map[t] for t in tags if t in self.merge_map ])
            self.merged_tags[arid] = list(set(m_tags))
        with open(os.path.join(self.data_path, "remerged_tags.json"), "w") as f:
            json.dump(self.merged_tags, f)

    def collect_merged_tag_stats(self):
        with open(os.path.join(self.data_path, "remerged_tags.json")) as f:
            self.merged_tags = json.load(f)
        self.tag_stats = { }
        self.tags = { }
        for _tags in self.merged_tags.values():
            for _tag in _tags:
                if _tag not in self.tag_stats:
                    self.tag_stats[_tag] = 1
                else:
                    self.tag_stats[_tag] += 1
        self.tags = { _k: _v for _k, _v in sorted(self.tag_stats.items(), key=lambda x:x[1], reverse=True ) }
        self.total_tags = sum([_v for _v in self.tags.values()])


class GenreClassifier:

    def __init__(self, data_dir="/data"):
        self.data_dir = data_dir

    def preprocess_data(self, mels_file="mels.json", tags_file="tags.json"):
        with open(os.path.join(self.data_dir, mels_file)) as f:
            self.mels = json.load(f)
        with open(os.path.join(self.data_dir, tags_file)) as f:
            self.tags = json.load(f)
        X, y = [], []
        for id, mels in self.mels.items():
            if np.array(mels).shape == (128, 128):
                for tag in self.tags.get(id, []):
                    X.append(mels)
                    y.append(tag)
        X, y = np.array(X), np.array(y)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1)
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

    def train_model(self, mels_file="bbc_melbands.json", tags_file="remerged_tags.json"):
        X_train, X_test, y_train, y_test = self.preprocess_data(mels_file, tags_file)
        X_train = X_train.reshape(X_train.shape[0], 128, 1292, 1)
        X_test = X_test.reshape(X_test.shape[0], 128, 1292, 1)
        y_train_cat = to_categorical(y_train, num_classes=10)
        y_test_cat = to_categorical(y_test, num_classes=10)

        model = self.build_model()
        model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=200, batch_size=32)
        return model

if __name__ == "__main__":
    data_path = "/data"  # Update with actual path
    genre_classifier = GenreClassifier(data_path)
    trained_model = genre_classifier.train_model()
    print("Genre classification model trained successfully!")
