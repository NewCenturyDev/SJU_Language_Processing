import json
import os
import pymysql
import platform

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

PREPED_DATA_PATH = "./preprocessed/"
MAX_SEQ_LEN = 52

# DB 접속정보
MYSQL_SERVER = 'localhost' if platform.system() == 'Windows' else 'sju_nlp_database'
MYSQL_USER = 'sju_nlp'
MYSQL_PASSWORD = 'password'
MYSQL_SCHEMA = 'sju_nlp'

tokenizer = Tokenizer(num_words=8150, lower=True, split=" ", oov_token="~")


def load_train_data():
    # 데이터 로드 (DB에서)
    connection = pymysql.connect(
        host=MYSQL_SERVER, port=3306, user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_SCHEMA, autocommit=False
    )
    cursor = connection.cursor()
    sql_query = 'SELECT text, sentiment FROM sentence_train;'
    train_data = pd.read_sql(sql_query, connection)
    cursor.close()
    connection.close()
    return train_data


def save_prep_train_data(train_inputs, train_labels, clean_train_df, data_configs):
    # 디렉터리 없을 시 생성
    if not os.path.exists(PREPED_DATA_PATH):
        os.makedirs(PREPED_DATA_PATH)

    train_input_data = "train_input.npy"
    train_label_data = "train_label.npy"
    train_clean_data = "train_clean.csv"
    data_config_name = "data_configs.json"
    # 넘파이 배열을 바이너리로 저장
    np.save(PREPED_DATA_PATH + train_input_data, train_inputs)
    np.save(PREPED_DATA_PATH + train_label_data, train_labels)
    # 텍스트 CSV 저장
    clean_train_df.to_csv(PREPED_DATA_PATH + train_clean_data, index=False)
    # JSON 저장
    json.dump(data_configs, open(PREPED_DATA_PATH + data_config_name, "w"), ensure_ascii=False)


def one_hot_encode_label(sentiments):
    encoded_sentiment = []
    for sentiment in sentiments:
        if sentiment == -1:
            encoded_sentiment.append([1, 0, 0])
        elif sentiment == 0:
            encoded_sentiment.append([0, 1, 0])
        else:
            encoded_sentiment.append([0, 0, 1])
    return encoded_sentiment


def vectorize_words(clean_train_texts, train_data):
    clean_train_df = pd.DataFrame({"text": clean_train_texts, "sentiment": train_data["sentiment"]})
    # 정제된 데이터에 토크나이징 적용
    tokenizer.fit_on_texts(clean_train_texts)
    # 인덱스로 구성된 벡터로 변환
    text_sequences = tokenizer.texts_to_sequences(clean_train_texts)
    print(text_sequences[0])

    # 패딩 정의
    word_vocab = tokenizer.word_index
    word_vocab["<PAD>"] = 0
    # print(word_vocab)
    print("전체 단어 개수: ", len(word_vocab))

    # 단어 사전 및 개수 저장
    data_configs = {"vocab": word_vocab, "vocab_size": len(word_vocab)}

    # 데이터 길이 통일
    train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQ_LEN, padding="post")
    print("Shape of train data: ", train_inputs.shape)

    # 라벨 배열 저장
    train_labels = np.array(one_hot_encode_label(train_data["sentiment"]))
    print("Shape of label tensor: ", train_labels.shape)

    # 전처리 데이터를 파일시스탬에 저장
    # 넘파이 데이터, 라벨 배열 및 텍스트 csv, json 저장
    save_prep_train_data(train_inputs, train_labels, clean_train_df, data_configs)


def main():
    # 메인 함수
    # 데이터를 로딩
    dataset = load_train_data()
    # print("데이터 전처리 시작")
    # 전체 리뷰 데이터에 대한 데이터 전처리 수행
    # preprocessed_train_texts = []
    # for text in dataset["text"]:
    #     preprocessed_train_texts.append(preprocessing(text, remove_stopwords=True))
    # print("데이터 전처리 종료")

    print("데이터 벡터화 시작")
    vectorize_words(dataset['text'], dataset)
    print("데이터 벡터화 종료")
    tokenizer_json = tokenizer.to_json()
    json.dump(tokenizer_json, open(PREPED_DATA_PATH + 'tokenizer.json', "w"), ensure_ascii=False)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
