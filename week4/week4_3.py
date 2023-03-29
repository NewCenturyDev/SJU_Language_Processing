# import os

import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 로깅 라이브러리 세팅
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


def count_vectorizing(reviews):
    # 카운트 벡터화
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    train_data_features = vectorizer.fit_transform(reviews)
    return {
        'features': train_data_features,
        'vectorizer': vectorizer
    }


def split_data(train_data_features, sentiments):
    x = train_data_features
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


def build_rfc(splited_data):
    # 랜덤 포레스트 분류기를 구성하는 함수 (100개의 의사 결정 트리 사용)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(splited_data['x_train'], splited_data['y_train'])
    print("Accuracy: %f" % rfc.score(splited_data['x_test'], splited_data['y_test']))
    return rfc


def do_validation(test_reviews, vectorizer, rfc):
    test_data_features = vectorizer.transform(test_reviews)
    test_predict = rfc.predict(test_data_features)
    return test_predict


def save_result_to_csv(test_data, test_predicted):
    # if not os.path.exists(DATA_OUT_PATH):
    #     os.makedirs(DATA_OUT_PATH)

    answer_dataset = pd.DataFrame({'id': test_data['ids'], 'sentiment': test_predicted})
    answer_dataset.to_csv(DATA_OUT_PATH + 'bag_of_words_model.csv', index=False, quoting=3)


def main():
    # 메인 함수
    print("시작")
    # 학습 데이터를 로딩
    train_dataset = load_train_data()
    count_vector = count_vectorizing(train_dataset['reviews'])
    splited_reviews = split_data(count_vector['features'], train_dataset['sentiment'])
    rfc_model = build_rfc(splited_reviews)
    test_dataset = load_test_data()
    test_predicted = do_validation(test_dataset['reviews'], count_vector['vectorizer'], rfc_model)
    save_result_to_csv(test_dataset, test_predicted)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
