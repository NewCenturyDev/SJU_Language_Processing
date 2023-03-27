import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATA_IN_PATH = "~/Documents/SJU_Language_Processing/week3/data/preped/"
TRAIN_CLEAN_DATA = "train_clean.csv"
TEST_CLEAN_DATA = "test_clean.csv"
DATA_OUT_PATH = "~/Documents/SJU_Language_Processing/week4/data/"


def load(data_file_name):
    # 데이터 로드 (훈련 데이터 또는 테스트 데이터)
    data = pd.read_csv(DATA_IN_PATH + data_file_name)
    reviews = list(data['review'])
    sentiment = None
    id = None
    if data_file_name is TRAIN_CLEAN_DATA:
        sentiment = list(data['sentiment'])
    if data_file_name is TEST_CLEAN_DATA:
        id = list(data['id'])
    return {
        'reviews': reviews,
        'sentiment': sentiment,
        'id': id,
    }


def train_tf_idf_vectorizing(clean_train_reviews, sentiments):
    # 리뷰의 단어들에 대해서 단어사전을 만들고 TF-IDF를 측정하여 벡터화 및 행렬로 표기해주는 라이브러리 함수 호출
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
    # 전체 문장에 대한 특징 벡터 데이터 X 생성
    x = vectorizer.fit_transform(clean_train_reviews)

    # 학습과 검증 데이터셋 분리
    # 단어 사전 (행렬 테이블의 X축) 불러오기
    features = vectorizer.get_feature_names_out()
    # 랜덤 시드 및 테스트 데이터 비율 설정
    random_seed = 42
    test_split = 0.2
    y = np.array(sentiments)
    # 테스트 데이터와 학습 데이터를 분리 (x=벡터화된 텍스트 행렬, y= 각 행렬에 대한 분류 라벨)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=test_split, random_state=random_seed)
    return {
        'features': features,
        'vectorizer': vectorizer,
        'X_train': x_train,
        'X_eval': x_eval,
        'y_train': y_train,
        'y_eval': y_eval
    }


def test_tf_idf_vectorizing(clean_test_reviews, trained_vectorizer, trained_lgs):
    test_data_vectors = trained_vectorizer.transform(clean_test_reviews)
    test_predicted = trained_lgs.predict(test_data_vectors)
    print(test_predicted)
    return test_predicted


def save_result_to_csv(test_data, test_predicted):
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    answer_dataset = pd.DataFrame({'id': test_data['id'], 'sentiment': test_predicted})
    answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False, quoting=3)


def do_train(x_train, y_train):
    # 로지스틱 회귀 모델 선언 및 학습 수행 (결정경계선은 중립으로)
    lgs = LogisticRegression(class_weight="balanced")
    lgs.fit(x_train, y_train)
    return lgs


def do_validation(trained_lgs, x_eval, y_eval):
    # 테스트 데이터를 이용하여 정확도에 대한 평가 수행
    predicted = trained_lgs.predict(x_eval)
    print("Accuracy: %f" % trained_lgs.score(x_eval, y_eval))
    return predicted


def main():
    # 메인 함수
    print("시작")
    # 학습 데이터를 로딩
    train_dataset = load(TRAIN_CLEAN_DATA)
    vectorizing_result = train_tf_idf_vectorizing(train_dataset['reviews'], train_dataset['sentiment'])
    trained_model = do_train(vectorizing_result['X_train'], vectorizing_result['y_train'])
    test_predicted = do_validation(trained_model, vectorizing_result['X_eval'], vectorizing_result['y_eval'])
    eval_dataset = load(TEST_CLEAN_DATA)
    # 평가 데이터 TF-IDF 벡터화
    eval_predicted = test_tf_idf_vectorizing(eval_dataset['reviews'], vectorizing_result['vectorizer'], trained_model)
    save_result_to_csv(eval_dataset, eval_predicted)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
