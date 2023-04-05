import json
import os
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

DATA_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week3/data/"
PREPED_DATA_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week3/data/preped/"
MAX_SEQ_LEN = 174
tokenizer = Tokenizer()


def load():
    # 데이터 로드 및 파일 크기 찾기
    print("파일 크기 : ")
    megabyte = 1000000
    for file in os.listdir(DATA_PATH):
        if 'tsv' in file and 'zip' not in file:
            print(file.ljust(30) + str(round(os.path.getsize(DATA_PATH + file) / megabyte, 2)))
    train_data = pd.read_csv(DATA_PATH + "labeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
    return train_data


def do_remove_stopwords(words):
    # 영어 불용어 처리
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return words


def preprocessing(review, remove_stopwords=False):
    # 전처리 과정 (불용어 제거는 옵션)
    # HTML 태그 제거
    review_text = BeautifulSoup(review, "html5lib").get_text()

    # 비 알파벳 문자 제거
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # 대문자를 전부 소문자로 변환하고 공백 단위로 토크나이징
    words = review_text.lower().split()

    # 불용어 제거
    if remove_stopwords:
        words = do_remove_stopwords(words)
    # 불용어 필터링 처리 또는 처리하지 않은 단어 리스트를 다시 줄글로 합침
    clean_review = " ".join(words)
    return clean_review


def save_prep_train_data(train_inputs, train_labels, clean_train_df, data_configs):
    # 디렉터리 없을 시 생성
    if not os.path.exists(PREPED_DATA_PATH):
        os.makedirs(PREPED_DATA_PATH)

    train_input_data = "train_input.npy"
    train_label_data = "train_label.npy"
    train_clean_data = "train_clean.csv"
    data_config_name = "data_configs.json"
    # 넘파이 배열을 바이너리로 저장
    np.save(open(PREPED_DATA_PATH + train_input_data, "wb"), train_inputs)
    np.save(open(PREPED_DATA_PATH + train_label_data, "wb"), train_labels)
    # 텍스트 CSV 저장
    clean_train_df.to_csv(PREPED_DATA_PATH + train_clean_data, index=False)
    # JSON 저장
    json.dump(data_configs, open(PREPED_DATA_PATH + data_config_name, "w"), ensure_ascii=False)


def save_prep_test_data(test_inputs, clean_test_df, test_id):
    # 디렉터리 없을 시 생성
    if not os.path.exists(PREPED_DATA_PATH):
        os.makedirs(PREPED_DATA_PATH)

    test_input_data = "test_input.npy"
    test_clean_data = "test_clean.csv"
    test_id_data = 'test_id.npy'

    # 넘파이 배열을 바이너리로 저장
    np.save(open(PREPED_DATA_PATH + test_input_data, "wb"), test_inputs)
    np.save(open(PREPED_DATA_PATH + test_id_data, 'wb'), test_id)
    # 텍스트 CSV 저장
    clean_test_df.to_csv(PREPED_DATA_PATH + test_clean_data, index=False)


def vectorize_words(clean_train_reviews, train_data):
    clean_train_df = pd.DataFrame({"review": clean_train_reviews, "sentiment": train_data["sentiment"]})
    # 정제된 데이터에 토크나이징 적용
    tokenizer.fit_on_texts(clean_train_reviews)
    # 인덱스로 구성된 벡터로 변환
    text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
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
    train_labels = np.array(train_data["sentiment"])
    print("Shape of label tensor: ", train_labels.shape)

    # 전처리 데이터를 파일시스탬에 저장
    # 넘파이 데이터, 라벨 배열 및 텍스트 csv, json 저장
    save_prep_train_data(train_inputs, train_labels, clean_train_df, data_configs)


def prep_test_data():
    test_data = pd.read_csv(DATA_PATH + "testData.tsv", header=0, delimiter="\t", quoting=3)

    # 평가 데이터 전처리 수행
    clean_test_reviews = []
    for review in test_data["review"]:
        clean_test_reviews.append(preprocessing(review, remove_stopwords=True))

    clean_test_df = pd.DataFrame({"review": clean_test_reviews, "id": test_data["id"]})
    test_id = np.array(test_data["id"])

    # 토크나이징 객체를 새롭게 만들면 인덱스가 바뀌어 적절하게 평가하거나 테스트 할 수 없음
    text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
    test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQ_LEN, padding="post")

    # 테스트 데이터 저장
    save_prep_test_data(test_inputs, clean_test_df, test_id)


def main():
    # 메인 함수
    print("시작")
    # 데이터를 로딩
    dataset = load()
    # 전체 리뷰 데이터에 대한 데이터 전처리 수행
    preprocessed_train_reviews = []
    for review in dataset["review"]:
        preprocessed_train_reviews.append(preprocessing(review, remove_stopwords=True))
    print(preprocessed_train_reviews[0])

    vectorize_words(preprocessed_train_reviews, dataset)
    prep_test_data()


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
