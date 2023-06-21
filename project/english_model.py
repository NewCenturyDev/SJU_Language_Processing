import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Convolution1D, MaxPooling1D, Activation
from keras.models import Sequential
from keras.optimizers import Adam

# noinspection DuplicatedCode
matplotlib.use('TkAgg')

# 데이터 입출력 경로
DATA_IN_PATH = "./preprocessed/"
DATA_OUT_PATH = "./result/"
# noinspection DuplicatedCode
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_INPUT_DATA = "test_input.npy"
TEST_ID_DATA = "test_id.npy"
DATA_CONFIGS = 'data_configs.json'

# 랜덤 시드 고정
SEED_NUM = 77
tf.random.set_seed(SEED_NUM)

# Check if a GPU is available
if tf.test.is_gpu_available():
    print("GPU is available. Setting TensorFlow to use GPU.")
    # Set TensorFlow to use GPU
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
else:
    print("GPU is not available. Setting TensorFlow to use CPU.")


def load_train_data():
    # RNN 학습 데이터 로드 (numpy 배열)
    train_input = np.load(DATA_IN_PATH+TRAIN_INPUT_DATA, 'r')
    train_label = np.load(DATA_IN_PATH+TRAIN_LABEL_DATA, 'r')
    prepro_configs = json.load(open(DATA_IN_PATH+DATA_CONFIGS, 'r', encoding='utf-8'))

    return {
        'train_input': train_input,
        'train_label': train_label,
        'prepro_configs': prepro_configs
    }


# def load_test_data():
#     # RNN 테스트 데이터 로드 (numpy 배열)
#     test_input = np.load(DATA_IN_PATH + TEST_INPUT_DATA, 'r')
#     test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'), allow_pickle=True)
#     return {
#         'test_input': pad_sequences(test_input, maxlen=test_input.shape[1]),
#         'test_id': test_id
#     }


def plot_graph(history, string):
    # 그래프 시각화 함수
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def setup_model_and_train(train_input, train_label):
    # 모델 하이퍼 파라미터 설정 및 모델 인스턴스 생성
    model_name = 'nlp_classifier_en'
    batch_size = 512
    num_epoches = 5
    valid_split = 0.25
    # max_len = train_input.shape[1]
    # https://www.kaggle.com/code/ngyptr/multi-class-classification-with-lstm
    kargs = {
        'model_name': model_name,
        'vocab_size': 1920,
        'embedding_size': 32,
        'dropout_rate': 0.5,
        'cnn_dimension': 128,
        'lstm_dimension': 128,
        'dense_dimension': 128,
        'output_dimension': 3
    }
    # model = RNNClassifier(**kargs)

    # 모델 컴파일
    # noinspection DuplicatedCode
    model = Sequential()
    model.add(Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_dimension']))
    model.add(Convolution1D(kargs['cnn_dimension'], kargs['output_dimension'], padding='same', strides=1))
    model.add(Activation('leaky_relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(kargs['lstm_dimension'], return_sequences=True, activation='leaky_relu'))
    model.add(LSTM(kargs['lstm_dimension'], activation='leaky_relu'))
    model.add(Dropout(kargs['dropout_rate']))
    model.add(Dense(kargs['dense_dimension'], activation='leaky_relu'))
    model.add(Dropout(kargs['dropout_rate']))
    model.add(Dense(kargs['output_dimension'], activation='softmax'))
    Adam(learning_rate=0.00146, name='Adam')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # 콜백 선언 - 오버피팅을 막기 위한 earlystop 추가
    # 0.0001 이상의 정확도 상승이 1회 이상 없으면 중지
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001)

    # noinspection DuplicatedCode
    checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_dir):
        print('{} -- Folder already exists \n'.format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print('{} -- Folder create complete \n'.format(checkpoint_dir))

    cp_callback = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True
    )

    # 모델 학습
    history = model.fit(
        train_input, train_label, batch_size=batch_size, epochs=num_epoches, validation_split=valid_split,
        callbacks=[earlystop_callback, cp_callback]
    )
    return {
        'model': model,
        'history': history,
        'batch_size': batch_size
    }


# noinspection DuplicatedCode
def main():
    # 메인 함수
    print("전처리된 데이터 학습 시작")
    # 학습 데이터를 로딩
    train_dataset = load_train_data()

    model_result = setup_model_and_train(
        train_dataset['train_input'], train_dataset['train_label']
    )
    print("전처리된 데이터 학습 종료")

    # 모델 정확도 시각화
    plot_graph(model_result['history'], 'accuracy')
    plot_graph(model_result['history'], 'loss')


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
