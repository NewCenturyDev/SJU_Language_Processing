# import os

import numpy as np
import pandas as pd
import logging

from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 로깅 라이브러리 세팅
# noinspection DuplicatedCode
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_IN_PATH = "~/Documents/SJU_Language_Processing/week3/data/preped/"
TRAIN_CLEAN_DATA = "train_clean.csv"
TEST_CLEAN_DATA = "test_clean.csv"
DATA_OUT_PATH = "~/Documents/SJU_Language_Processing/week4/data/"


WORD2VEC_PARAMS = {
    'num_features': 300,
    'min_word_count': 40,
    'num_workers': 4,
    'context': 10,
    'downsampling': 1e-3
}


def load_train_data():
    train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
    # Word2Vec은 단어로 표현된 리스트를 입력값으로 사용
    reviews = list(train_data['review'])
    return {
        'reviews': reviews,
        'sentiment': train_data['sentiment']
    }


def load_test_data():
    test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
    return {
        'reviews': list(test_data['review']),
        'ids': list(test_data['id'])
    }


def split_data(data_vecs, sentiments):
    x = data_vecs
    y = np.array(sentiments)

    random_seed = 42
    test_split = 0.2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_seed)
    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }


def build_model(clean_train_reviews):
    # word2vec 모델을 구성 및 빌드하는 함수
    # 원래 데이터의 문장들을 공백 단위로 쪼개서 삽입
    sentences = []
    for review in clean_train_reviews:
        sentences.append(review.split())
    model = word2vec.Word2Vec(
        sentences, workers=WORD2VEC_PARAMS['num_workers'],
        vector_size=WORD2VEC_PARAMS['num_features'], min_count=WORD2VEC_PARAMS['min_word_count'],
        window=WORD2VEC_PARAMS['context'], sample=WORD2VEC_PARAMS['downsampling']
    )
    model_name = '{}features_{}minwords_{}context'.format(
        WORD2VEC_PARAMS['num_features'], WORD2VEC_PARAMS['min_word_count'], WORD2VEC_PARAMS['context']
    )
    model.save(model_name)
    return model


def get_features(words, model, num_features):
    # 한 리뷰에 대해 전체 단어의 평균값을 계산하는 함수
    # words = 한 리뷰를 구성하는 단어의 모음
    # model = 바로 위에서 구성한 word2vec 모델
    # num_features = 모델을 구성할 때 정했던 벡터의 차원 수

    # 출력 벡터 초기화
    feature_vector = np.zeros(num_features, dtype=np.float32)
    num_words = 0

    # 어휘사전 준비
    index2word_set = set(model.wv.index_to_key)

    for w in words:
        if w in index2word_set:
            # 사전에 해당하는 단어에 대해 단어 벡터에 카운트 증가
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[w])

    # 문장의 단어 수로 나누어 단어 벡터의 평균값을 문장 벡터로 취급
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector


def get_dataset(reviews, model, num_features):
    # 전체 리뷰에 대해 각 리뷰의 평균 벡터를 계산하는 함수
    dataset = list()

    for review in reviews:
        dataset.append(get_features(review, model, num_features))

    review_feature_vecs = np.stack(dataset)
    return review_feature_vecs


def save_result_to_csv(test_data, test_predicted):
    # if not os.path.exists(DATA_OUT_PATH):
    #     os.makedirs(DATA_OUT_PATH)

    answer_dataset = pd.DataFrame({'id': test_data['ids'], 'sentiment': test_predicted})
    answer_dataset.to_csv(DATA_OUT_PATH + 'bag_of_words_lgs.csv', index=False, quoting=3)


def test_word2vec_vectorizing(clean_test_review, model, num_features):
    sentences = []
    for review in clean_test_review:
        sentences.append(review.split())
    test_data_vecs = get_dataset(sentences, model, num_features)
    return test_data_vecs


def do_train(splited_data):
    # 로지스틱 회귀 모델 선언 및 학습 수행 (결정경계선은 중립으로)
    lgs = LogisticRegression(class_weight="balanced")
    lgs.fit(splited_data['x_train'], splited_data['y_train'])
    print("Accuracy: %f" % lgs.score(splited_data['x_test'], splited_data['y_test']))
    return lgs


def do_validation(trained_lgs, test_data_vecs):
    return trained_lgs.predict(test_data_vecs)


def main():
    # 메인 함수
    print("시작")
    # 학습 데이터를 로딩
    train_dataset = load_train_data()
    word2vec_model = build_model(train_dataset['reviews'])
    train_data_vec = get_dataset(train_dataset['reviews'], word2vec_model, WORD2VEC_PARAMS['num_features'])
    splited_reviews = split_data(train_data_vec, train_dataset['sentiment'])
    trained_lgs_model = do_train(splited_reviews)

    eval_dataset = load_test_data()
    test_data_vecs = test_word2vec_vectorizing(eval_dataset['reviews'], word2vec_model, WORD2VEC_PARAMS['num_features'])
    test_predicted = do_validation(trained_lgs_model, test_data_vecs)
    save_result_to_csv(eval_dataset, test_predicted)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
