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
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Dropout, Lambda
from tensorflow.keras.layers import GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel, BertConfig
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')

# Define model layers for fine-tuning
class BertBiLSTMModel(Model):
    def __init__(self, bert_model, lstm_size=128, num_classes=10):
        super(BertBiLSTMModel, self).__init__()
        self.bert = bert_model
        for layer in self.bert.layers:
            layer.trainable = False  # Freeze BERT layers
        self.bilstm = Bidirectional(LSTM(lstm_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        self.global_max_pool = GlobalMaxPool1D()
        self.dense1 = Dense(lstm_size, activation='relu')
        self.dropout = Dropout(0.2)
        self.classifier = Dense(num_classes, activation='sigmoid')  # Adjust num_classes as needed

    def call(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.bilstm(bert_outputs.last_hidden_state)
        x = self.global_max_pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.classifier(x)


def clean_text(temp):
    temp=re.sub("@\S+", " ", temp)
    temp=re.sub("https*\S+", " ", temp)
    temp=re.sub("#\S+", " ", temp)
    temp=re.sub("\'\w+", '', temp)
    temp=re.sub(r'\w*\d+\w*', '', temp)
    temp=re.sub('\s{2,}', " ", temp)
    return temp.strip()


class BertBiLSTMClassifier():
    def __init__(self):
        self.dir = "./"
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
        self.sentences = self.ds['text']
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
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.model = BertBiLSTMModel(self.bert_model, self.input_size, self.unique.shape[0])
        self.model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=[self.train_accuracy])
        print(self.model.summary())

    def fit(self, epochs, batch_size=32, val_size=0.1):
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_ids, self.target, self.attention_masks))
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            num_batches = 0
            pbar = tqdm(self.train_dataset, total=len(self.train_dataset), desc=f"Epoch {epoch + 1}", unit="batch")
            for step, (texts, labels, mask) in enumerate(self.train_dataset):

                # Convert labels to TensorFlow tensors if needed
                labels = tf.convert_to_tensor(labels)

                # Run training step
                loss = self.train_step(tf.expand_dims(texts, axis=0), tf.expand_dims(mask, axis=0), tf.expand_dims(labels, axis=0))

                epoch_loss += loss.numpy()
                num_batches += 1

                pbar.update(1)

                pbar.set_postfix({
                    "Loss": epoch_loss / num_batches,
                    "Accuracy": self.train_accuracy.result().numpy()
                })

            self.train_accuracy.reset_state()

    @tf.function
    def train_step(self, input_ids, attention_mask, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(input_ids, attention_mask)
            loss = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_accuracy.update_state(labels, predictions)
        return loss
