import re
import os
import json
import pickle
import numpy as np

import tensorflow as tf
from konlpy.tag import Okt
from keras import layers
from keras_preprocessing.sequence import pad_sequences

PATH = 'Models/cnn_classifier_kr/'
DATA_CONFIGS = 'data_configs.json'
TOKENIZER_PATH = 'tokenizer.pkl'
SAVE_FILE_NM = 'weights_cpu.h5'  # 저장된 best model 이름
JVM_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home/jre/lib/server/libjvm.dylib"

MAX_SEQUENCE_LENGTH = 8  # 문장 최대 길이

with open(os.path.join(PATH, TOKENIZER_PATH), 'rb') as fp:
    tokenizer = pickle.load(fp)


def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)
    word_review = okt.morphs(review_text, stem=True)
    if remove_stopwords:
        word_review = [token for token in word_review if token not in stop_words]
    return word_review


prepro_configs = json.load(open(PATH + DATA_CONFIGS, 'r', encoding='utf-8'))

kargs = {'model_name': PATH,
         'vocab_size': prepro_configs['vocab_size'],
         'embedding_size': 128,
         'num_filters': 100,
         'dropout_rate': 0.5,
         'hidden_dimension': 250,
         'output_dimension': 1}


class CNNClassifier(tf.keras.Model):

    def __init__(self, **kargs):
        super(CNNClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                          output_dim=kargs['embedding_size'])
        self.conv_list = [layers.Conv1D(filters=kargs['num_filters'],
                                        kernel_size=kernel_size,
                                        padding='valid',
                                        activation=tf.keras.activations.relu,
                                        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
                          for kernel_size in [3, 4, 5]]
        self.pooling = layers.GlobalMaxPooling1D()
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.fc1 = layers.Dense(units=kargs['hidden_dimension'],
                                activation=tf.keras.activations.relu,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
        self.fc2 = layers.Dense(units=kargs['output_dimension'],
                                activation=tf.keras.activations.sigmoid,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


model = CNNClassifier(**kargs)

model.build(input_shape=(kargs['vocab_size'], kargs['embedding_size']))

model.load_weights(os.path.join(PATH, SAVE_FILE_NM))

with open(os.path.join(PATH, TOKENIZER_PATH), 'rb') as fp:
    tokenizer = pickle.load(fp)

stop_words = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt(JVM_HOME)


def tokenize_func(text):
    clean_review = []

    for review in text:  # 코드 실행 시간이 긴 관계로 tqdm을 추가함
        # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
        if type(review) == str:
            clean_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
        else:
            clean_review.append([])  # string이 아니면 비어있는 값 추가

    input_data = tokenizer.texts_to_sequences(clean_review)
    input_data = pad_sequences(input_data, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return input_data


def predict_func(text):
    tokenize_padseq_text = tokenize_func(text)
    output = model.predict(tokenize_padseq_text)
    output = np.where(output > 0.5, 1, 0)
    return output
