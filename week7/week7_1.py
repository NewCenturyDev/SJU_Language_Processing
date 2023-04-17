import pandas as pd
import numpy as np
import re
import json

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


DATA_IN_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/'
DATA_OUT_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/preped/'


def load_train_data():
    return pd.read_csv(DATA_IN_PATH + 'train.csv', encoding='utf-8')


def load_test_data():
    return pd.read_csv(DATA_IN_PATH + 'test.csv', encoding='utf-8')


def preprocess_data(train_data):
    # 학습 데이터에서 긍부정 데이터의 라벨 불러오기
    train_pos_data = train_data.loc[train_data['is_duplicate'] == 1]
    train_neg_data = train_data.loc[train_data['is_duplicate'] == 0]

    # 긍정/부정 분류간 데이터 차이
    class_difference = len(train_neg_data) - len(train_pos_data)
    # 클래스간 데이터 차이 비율
    sample_frac = 1 - (class_difference / len(train_neg_data))
    # 비율만큼 샘플링 (긍정과 부정 데이터의 개수 및 비율을 동등하게 맞춰 줌)
    train_neg_data = train_neg_data.sample(frac=sample_frac)
    print('중복 질문 개수: {}'.format(len(train_pos_data)))
    print('중복이 아닌 질문 개수: {}'.format(len(train_neg_data)))

    # 형 변환
    train_data = pd.concat([train_neg_data, train_pos_data])
    questions1 = [str(s) for s in train_data['question1']]
    questions2 = [str(s) for s in train_data['question2']]

    # 필터 설정 (길이가 지나치게 긴 극단적인 이상치 제거 및 문장부호 제거, 소문자화)
    max_sequences_length = 31
    filters = '([~.,!?"\':;)(])'
    change_filter = re.compile(filters)

    # 정규화 수행 (문장부호 제거, 소문자화)
    filtered_questions1 = list()
    filtered_questions2 = list()
    for q in questions1:
        filtered_questions1.append(re.sub(change_filter, '', q).lower())
    for q in questions2:
        filtered_questions2.append(re.sub(change_filter, '', q).lower())

    # 토크나이징 수행 (q1, q2 합쳐서 토크나이징 객체를 만들고, 토크나이징 자체는 따로 수행)
    # 합쳐서 토크나이징 객체를 만드는 이유는, 단어사전이 q1과 q2에 있는 모든 단어를 커버할 수 있도록 하기 위함이다
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(filtered_questions1+filtered_questions2)

    questions1_sequence = tokenizer.texts_to_sequences(filtered_questions1)
    questions2_sequence = tokenizer.texts_to_sequences(filtered_questions2)

    # 패딩 설정하여 벡터의 길이를 동등하게 맞춰 줌 (max_length에 맞게)
    q1_data = pad_sequences(questions1_sequence, maxlen=max_sequences_length, padding='post')
    q2_data = pad_sequences(questions2_sequence, maxlen=max_sequences_length, padding='post')

    # 단어사전
    word_vocab = {}
    word_vocab = tokenizer.word_index
    word_vocab['<PAD>'] = 0

    # 라벨을 np 어레이로 변환
    labels = np.array(train_data['is_duplicate'], dtype=int)

    print('Shape of question 1 data: {}'.format(q1_data.shape))
    print('Shape of question 2 data: {}'.format(q2_data.shape))
    print('Shape of label: {}'.format(labels.shape))
    print('Words in index: {}'.format(len(word_vocab)))

    # 전처리된 벡터 데이터, 라벨, 단어 사전 등 설정을 파일로 저장
    data_configs = {'vocab': word_vocab, 'vocab_size': len(word_vocab)}
    train_q1_data = 'train_q1.npy'
    train_q2_data = 'train_q2.npy'
    train_label_data = 'train_label.npy'
    data_configs = 'data_configs.json'

    np.save(open(DATA_OUT_PATH + train_q1_data, 'wb'), q1_data)
    np.save(open(DATA_OUT_PATH + train_q2_data, 'wb'), q2_data)
    np.save(open(DATA_OUT_PATH + train_label_data, 'wb'), labels)
    json.dump(data_configs, open(DATA_OUT_PATH + data_configs, 'w'))
    return tokenizer


def preprocess_test_data(test_data, tokenizer):
    # 필터 설정 (길이가 지나치게 긴 극단적인 이상치 제거 및 문장부호 제거, 소문자화)
    max_sequences_length = 31
    filters = '([~.,!?"\':;)(])'
    change_filter = re.compile(filters)

    # 형 변환
    valid_ids = [type(x) == int for x in test_data['test_id']]
    test_data = test_data[valid_ids].drop_duplicates()
    test_questions1 = [str(s) for s in test_data['question1']]
    test_questions2 = [str(s) for s in test_data['question2']]

    # 정규화 수행 (문장부호 제거, 소문자화)
    filtered_questions1 = list()
    filtered_questions2 = list()
    for q in test_questions1:
        filtered_questions1.append(re.sub(change_filter, '', q).lower())
    for q in test_questions2:
        filtered_questions2.append(re.sub(change_filter, '', q).lower())

    questions1_sequence = tokenizer.texts_to_sequences(filtered_questions1)
    questions2_sequence = tokenizer.texts_to_sequences(filtered_questions2)

    # 패딩 설정하여 벡터의 길이를 동등하게 맞춰 줌 (max_length에 맞게)
    test_q1_data = pad_sequences(questions1_sequence, maxlen=max_sequences_length, padding='post')
    test_q2_data = pad_sequences(questions2_sequence, maxlen=max_sequences_length, padding='post')

    # ID 추출
    test_id = np.array(test_data['test_id'])

    print('Shape of question 1 data: {}'.format(test_q1_data.shape))
    print('Shape of question 2 data: {}'.format(test_q2_data.shape))
    print('Shape of ids: {}'.format(test_id.shape))

    # 파일에 쓰기
    np.save(open(DATA_OUT_PATH+'test_q1.npy', 'wb'), test_q1_data)
    np.save(open(DATA_OUT_PATH+'test_q2.npy', 'wb'), test_q2_data)
    np.save(open(DATA_OUT_PATH+'test_id.npy', 'wb'), test_id)


data = load_train_data()
tkn = preprocess_data(data)
preprocess_test_data(load_test_data(), tkn)
