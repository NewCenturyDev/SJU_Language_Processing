import tensorflow as tf
import numpy as np
import os
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from week11_2 import *

# seq2seq.py


DATA_IN_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week11/data/"
DATA_OUT_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week11/result/'
TRAIN_INPUT = "train_inputs.npy"
TRAIN_OUTPUT = "train_outputs.npy"
TRAIN_TARGETS = "train_targets.npy"
DATA_CONFIGS = "data_configs.json"

SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# 전처리된 데이터 로드
index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUT, 'rb'))
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUT, 'rb'))
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
print(len(index_inputs), len(index_outputs), len(index_targets))

# 모델 구성에 필요한 값 선언
MODEL_NAME = 'seq2seq_kor'
BATCH_SIZE = 2
MAX_SEQUENCE = 25
EPOCH = 30
UNITS = 1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1

char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']
std_index = prepro_configs['std_symbol']
end_index = prepro_configs['end_symbol']
vocab_size = prepro_configs['vocab_size']

