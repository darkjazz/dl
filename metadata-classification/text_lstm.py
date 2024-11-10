import os
import re
import joblib
import json
import pandas as pd
import textdistance as td
import numpy as np
from progressbar import ProgressBar
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Dropout, Lambda
from tensorflow.keras.layers import GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel, BertConfig


class LstmLabelClassifier:
    def __init__(self):
        self.dir = os.path.join(os.path.expanduser('~'), "io", "tagds")
        self.source_file = "lstm-src-ds.csv"
        self.ds_file = "lstm-ds.csv"
        self.sample_map = {
            "functional": 0.05,
            "karaoke": 0.05,
            "music": 0.05,
            "religious": 0.6,
            "sfx": 0.1,
            "speech": 0.1
        }
        self.len_limit = 512
        self.random_state = 73
        self.test_size = 0.1

    def preprocess(self):
        self.ds = pd.read_csv(os.path.join(self.dir, self.source_file), header=None, error_bad_lines=False)
        self.ds = self.ds.rename(columns={0:"text", 1: "label"})
        self.ds["text"] = self.ds["text"].apply(self.cleanup)
        self.ds["length"] = self.ds['text'].apply(len)
        self.ds = self.ds.drop(self.ds.loc[(self.ds["length"] > self.len_limit)].index)
        self.ds.dropna(inplace=True)
        self.describe_class_distribution()
        self.ds = self.ds.sort_values(["text"])
        self.ds = self.ds.drop_duplicates(subset=["text"])
        self.describe_class_distribution()
        for _lbl, _frc in self.sample_map.items():
            _df = self.ds.loc[(self.ds["label"] == _lbl)].sample(frac=1.0-_frc, random_state=self.random_state)
            self.ds = self.ds.drop(_df.index)
        self.ds = self.ds.sort_values(["text"])
        self.reduce()
        self.describe_class_distribution()
        self.ds.to_csv(os.path.join(self.dir, self.ds_file))

    def load_data(self):
        self.ds = pd.read_csv(os.path.join(self.dir, self.ds_file))
        self.unique = self.ds["label"].unique()
        self.unique.sort()
        self.vocab = set()
        self.ds['text'].str.lower().str.split().apply(self.vocab.update)
        # self.ds = self.ds.sample(frac=1)

    def prepare_data(self):
        self.tokenizer = Tokenizer(num_words=len(self.vocab), char_level=False, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.ds['text'])
        self.xtr, self.xte, self.ytr, self.yte = train_test_split(
            self.ds["text"],
            np.array([ self.encode_label(_l) for _l in self.ds["label"]]),
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.tr_seq = self.tokenizer.texts_to_sequences(self.xtr)
        self.tr_pad = pad_sequences(self.tr_seq, maxlen=self.len_limit, padding='post', truncating='post')
        self.te_seq = self.tokenizer.texts_to_sequences(self.xte)
        self.te_pad = pad_sequences(self.te_seq, maxlen=self.len_limit, padding='post', truncating='post')

    def build_model(self, emb_size=128, lstm_size=128):
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab), emb_size, input_length=self.tr_pad.shape[1]))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(lstm_size, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.unique.shape[0], activation="sigmoid"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(self.model.summary())

    def build_bi_model(self, emb_size, lstm_size):
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab), emb_size, input_length=self.tr_pad.shape[1]))
        self.model.add(Bidirectional(LSTM(lstm_size, return_sequences = False)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.unique.shape[0], activation="sigmoid"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(self.model.summary())

    def fit(self, epochs):
        self.early_stop = EarlyStopping(monitor='val_loss', patience=2)
        self.history = self.model.fit(self.tr_pad, self.ytr, epochs=epochs, validation_data=(self.te_pad, self.yte), callbacks=[self.early_stop])

    def encode_label(self, label):
        arr = ([0.0]*self.unique.shape[0])
        arr[list(self.unique).index(label)] = 1.0
        return arr

    def describe_class_distribution(self):
        self.class_counts = {}
        unique_sorted = list(self.ds["label"].unique())
        unique_sorted.sort()
        for _lbl in unique_sorted:
            _sum = sum([1 for _r in self.ds["label"] if _r == _lbl ])
            print(_lbl, _sum)
            self.class_counts[_lbl] = _sum
        print("----")

    def cleanup(self, text):
        return text.replace(" shorter than a minute", "").replace(" longer than half an hour", "").replace(" registered", "").strip()

    def reduce(self, sim_thresh=0.9):
        self.delete = []
        prev = ""
        bar = ProgressBar(max_value=len(self.ds))
        step = 0
        for _i, _r in self.ds.iterrows():
            _sim = td.damerau_levenshtein.normalized_similarity(_r.text, prev)
            if _sim > sim_thresh:
                self.delete.append(_i)
            prev = _r.text
            bar.update(step)
            step += 1
        bar.finish()
        self.ds = self.ds.drop(index=self.delete)

    def predict(self, model, filename):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.ds_file = filename
        self.load_data()
        self.tokenizer = joblib.load(os.path.join(self.model_dir, f"{model}.tokenizer")) #Tokenizer(num_words=len(self.vocab), char_level=False, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.ds['text'])
        self.pr_seq = self.tokenizer.texts_to_sequences(self.ds["text"])
        self.pr_pad = pad_sequences(self.pr_seq, maxlen=self.len_limit, padding='post', truncating='post')
        self.model = load_model(os.path.join(self.model_dir, f"{model}.h5"))
        self.predictions = self.model.predict(self.pr_pad)


def clean_text(temp):
    temp=re.sub("@\S+", " ", temp)
    temp=re.sub("https*\S+", " ", temp)
    temp=re.sub("#\S+", " ", temp)
    temp=re.sub("\'\w+", '', temp)
    temp=re.sub(r'\w*\d+\w*', '', temp)
    temp=re.sub('\s{2,}', " ", temp)
    return temp.strip()


class LstmBertClassifier:
    def __init__(self):
        self.dir = os.path.join(os.path.expanduser('~'), "io", "tagds")
        self.model_name = "bert-base-multilingual-uncased"
        self.input_ids=[]
        self.attention_masks=[]
        self.input_size = 128

    def encode_label(self, label):
        arr = ([0.0]*self.unique.shape[0])
        arr[list(self.unique).index(label)] = 1.0
        return arr

    def init_bert(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.config = BertConfig.from_pretrained(self.model_name,output_hidden_states=False)
        self.bert_model = TFBertModel.from_pretrained(self.model_name, config=self.config)

    def load_data(self, filename):
        self.ds = pd.read_csv(os.path.join(self.dir, filename))
        self.unique = self.ds["label"].unique()
        self.unique.sort()
        self.vocab = set()
        self.ds['text'] = self.ds['text'].str.lower().apply(clean_text)
        self.ds['text'].str.split().apply(self.vocab.update)
        self.sentences = self.ds['text'][:1000]
        self.target = np.array([ self.encode_label(_l) for _l in self.ds["label"]])

    def tokenize(self):
        prg = ProgressBar(max_value=len(self.sentences))
        for i, sent in enumerate(self.sentences):
            bert_inp=self.bert_tokenizer.encode_plus(sent, add_special_tokens = True,
                max_length=self.input_size, pad_to_max_length = True, return_attention_mask = True)
            prg.update(i)
            self.input_ids.append(bert_inp['input_ids'])
            self.attention_masks.append(bert_inp['attention_mask'])
        prg.finish()

        self.input_ids = np.asarray(self.input_ids)
        self.attention_masks = np.array(self.attention_masks)

    def create_model(self, lstm_size=64):
        embedding_layer = self.bert_model(self.input_ids, attention_mask=self.attention_masks)
        embeddings = tf.convert_to_tensor(embedding_layer.last_hidden_state)
        out = Bidirectional(LSTM(lstm_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embeddings)
        out = GlobalMaxPool1D()(out)
        out = Dense(lstm_size, activation='relu')(out)
        out = Dropout(0.2)(out)
        out = Dense(self.unique.shape[0], activation='sigmoid')(out)
        self.model = tf.keras.Model(inputs=[self.input_ids, self.attention_masks], outputs = out)
        for layer in self.model.layers[:3]:
            layer.trainable = False
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(self.model.summary())

    def fit(self, epochs, batch_size=32, val_size=0.1):
        self.xtr, self.xte, self.ytr, self.yte, self.tr_mask, self.te_mask = train_test_split(
            self.input_ids,
            self.target,
            self.attention_masks,
            test_size=val_size
        )
        self.history = self.model.fit(
            [self.xtr, self.tr_mask],
            self.ytr,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([self.xte, self.te_mask], self.yte),
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
        )

    def predict(self, model, filename):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path = os.path.join(os.path.expanduser('~'), "io", filename)
        with open(path) as f:
            self.ds = json.load(f)
        self.model = load_model(os.path.join(self.model_dir, f"{model}.h5"))
        self.predictions = self.model.predict(self.pr_pad)
