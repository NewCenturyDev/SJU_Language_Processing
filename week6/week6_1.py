
import os
import json

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from CNNClassifier import CNNClassifier
import numpy as np
import pandas as pd

# noinspection DuplicatedCode
matplotlib.use('TkAgg')

# 데이터 입출력 경로
DATA_IN_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing//week3/data/preped/"
DATA_OUT_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing//week6/data/"
# noinspection DuplicatedCode
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_INPUT_DATA = "test_input.npy"
TEST_ID_DATA = "test_id.npy"
DATA_CONFIGS = 'data_configs.json'

# 랜덤 시드 고정
SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)


def load_train_data():
    # CNN 학습 데이터 로드 (numpy 배열)
    train_input = np.load(DATA_IN_PATH+TRAIN_INPUT_DATA, 'r')
    train_label = np.load(DATA_IN_PATH+TRAIN_LABEL_DATA, 'r')
    prepro_configs = json.load(open(DATA_IN_PATH+DATA_CONFIGS, 'r', encoding='utf-8'))

    return {
        'train_input': train_input,
        'train_label': train_label,
        'prepro_configs': prepro_configs
    }


def load_test_data():
    # CNN 테스트 데이터 로드 (numpy 배열)
    test_input = np.load(DATA_IN_PATH + TEST_INPUT_DATA, 'r')
    test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'), allow_pickle=True)
    return {
        'test_input': pad_sequences(test_input, maxlen=test_input.shape[1]),
        'test_id': test_id
    }


def plot_graph(history, string):
    # 그래프 시각화 함수
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def setup_model_and_train(train_input, train_label, prepro_configs):
    # 모델 하이퍼 파라미터 설정 및 모델 인스턴스 생성
    model_name = 'cnn_classifier_en'
    batch_size = 512
    num_epoches = 5
    valid_split = 0.1
    # max_len = train_input.shape[1]
    kargs = {
        'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
        'embedding_size': 128,
        'num_filters': 100,
        'dropout_rate': 0.5,
        'hidden_dimension': 250,
        'output_dimension': 1,
    }
    model = CNNClassifier(**kargs)

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

    # 콜백 선언 - 오버피팅을 막기 위한 earlystop 추가
    # 0.0001 이상의 정확도 상승이 2회 이상 없으면 중지
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)

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


def do_validation(model, test_input, batch_size):
    save_file_name = 'weights.h5'
    model.load_weights(os.path.join(DATA_OUT_PATH, model.name, save_file_name))
    predictions = model.predict(test_input, batch_size=batch_size)
    predictions = predictions.squeeze(-1)
    return predictions


def save_result_to_csv(test_id, test_predicted):
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    answer_dataset = pd.DataFrame({'id': list(test_id), 'sentiment': list(test_predicted)})
    answer_dataset.to_csv(DATA_OUT_PATH + 'movie_review_result_cnn.csv', index=False, quoting=3)


# noinspection DuplicatedCode
def main():
    # 메인 함수
    print("시작")
    # 학습 데이터를 로딩
    train_dataset = load_train_data()

    model_result = setup_model_and_train(
        train_dataset['train_input'], train_dataset['train_label'], train_dataset['prepro_configs']
    )

    plot_graph(model_result['history'], 'accuracy')
    plot_graph(model_result['history'], 'loss')

    test_dataset = load_test_data()
    test_predicted = do_validation(model_result['model'], test_dataset['test_input'], model_result['batch_size'])
    save_result_to_csv(test_dataset['test_id'], test_predicted)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
