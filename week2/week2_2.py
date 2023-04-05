import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

matplotlib.use('TkAgg')


def download_data():
    # 데이터 다운로드
    return tf.keras.utils.get_file(
        fname="imdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True
    )


def data_load(directory):
    # 텍스트 데이터를 불러와 데이터 프레임을 생성한다
    data = {"review": []}
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r", encoding="utf-8") as file:
            data["review"].append(file.read())
    return pd.DataFrame.from_dict(data)


def data_labeling(directory):
    # 긍정적인 텍스트 데이터와 부정적인 텍스트 데이터를 라벨링한다
    postive_df = data_load(os.path.join(directory, "pos"))
    postive_df["sentiment"] = 1
    negative_df = data_load(os.path.join(directory, "neg"))
    negative_df["sentiment"] = 0
    return pd.concat([postive_df, negative_df])


def visualize_def(review_len_by_token, review_len_by_eumjeol):
    # 그래프의 이미지 사이즈 선언
    plt.figure(figsize=(12, 5))
    # 히스토그램 선언
    # bins: 히스토그램 값들의 범위
    # range: x축 값의 범위
    # alpha: 그래프 색상 투명도
    # color: 그래프 색상
    # label: 그래프에 대한 라벨
    plt.hist(review_len_by_token, bins=50, alpha=0.5, color="r", label="word")
    plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color="b", label="alphabet")
    # y 스케일 조정 - 로그스케일, 넘치는 부분은 자름
    plt.yscale("log")
    plt.title("Review Length Histogram")
    plt.xlabel("Review Length")
    plt.ylabel("Number of Reviews")
    plt.show()


def do_edf_statistic(review_len_by_token):
    print("문장 최대길이: {}".format(np.max(review_len_by_token)))
    print("문장 최소길이: {}".format(np.min(review_len_by_token)))
    print("문장 평균길이: {}".format(np.mean(review_len_by_token)))
    print("문장 길이 표준편차: {}".format(np.std(review_len_by_token)))
    print("문장 중간길이: {}".format(np.median(review_len_by_token)))
    # 사분위의 경우 1~100 스케일
    print("제 1 사분위 길이: {}".format(np.percentile(review_len_by_token, 25)))
    print("제 3 사분위 길이: {}".format(np.percentile(review_len_by_token, 75)))

    # 토큰 개수에 대한 박스 플롯 시각화
    plt.figure(figsize=(12, 5))
    plt.boxplot([review_len_by_token], labels=["token"], showmeans=True)
    plt.show()

    # 음절 개수에 대한 박스 플롯 시각화
    plt.figure(figsize=(12, 5))
    plt.boxplot([review_len_by_token], labels=["Eumjeol"], showmeans=True)
    plt.show()


def do_edf(train_df):
    # 학습 데이터 셋의 리뷰 텍스트를 리스트로 변환
    reviews = list(train_df["review"])
    # 배열 안의 리뷰 텍스트를 띄어쓰기 단위로 토크나이징
    tokenized_reviews = [r.split() for r in reviews]
    # 토크나이징 된 리뷰들의 리스트에 대해 문장 각각의 토큰 개수를 측정
    review_len_by_token = [len(t) for t in tokenized_reviews]
    # 각 리뷰 텍스트 토큰들의 음절 길이 측정 (토크나이즈 된 것들을 붙여서 측정)
    review_len_by_eumjeol = [len(s.replace(" ", "")) for s in reviews]
    # 리뷰들의 토큰 및 음절 분포를 시각화
    visualize_def(review_len_by_token, review_len_by_eumjeol)
    do_edf_statistic(review_len_by_token)


def do_wordcloud(train_df):
    # 단어 빈도 시각화
    wordcloud = WordCloud(
        stopwords=STOPWORDS, background_color="black", width=800, height=600
    ).generate(
        "".join(train_df["review"])
    )
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def do_seaborn(train_df):
    # 데이터 라벨 분포
    sentiment = train_df["sentiment"].value_counts()
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(sentiment)
    plt.show()


def main():
    # 메인 함수
    print("시작")
    # 데이터를 다운로드
    dataset = download_data()
    print("다운로드 완료")
    # 학습 데이터와 테스트 데이터를 로드 및 라벨링
    train_df = data_labeling(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = data_labeling(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    print("학습 데이터와 테스트 데이터를 로드 완료")
    # 학습 데이터를 확인
    train_df.head()
    # EDF 수행을 위한 팩터 측정 및 시각화
    do_edf(train_df)
    do_wordcloud(train_df)
    do_seaborn(train_df)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
