import os
import re

import matplotlib
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

DATA_PATH = "/week3/data/"


def load():
    # 데이터 로드 및 파일 크기 찾기
    print("파일 크기 : ")
    megabyte = 1000000
    for file in os.listdir(DATA_PATH):
        if 'tsv' in file and 'zip' not in file:
            print(file.ljust(30) + str(round(os.path.getsize(DATA_PATH + file) / megabyte, 2)))
    train_data = pd.read_csv(DATA_PATH + "labeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
    return train_data


def print_data_size(train_data):
    # 학습 데이터 확인 및 데이터 갯수 산출
    train_data.head()
    print("전체 학습데이터의 개수: {}".format(len(train_data)))


def print_statistics(name, counts):
    # 통계 데이터 계산 및 출력
    print("{} 최대 값: {}".format(name, np.max(counts)))
    print("{} 최소 값: {}".format(name, np.min(counts)))
    print("{} 평균 값: {:2f}".format(name, np.mean(counts)))
    print("{} 표준편차: {:2f}".format(name, np.std(counts)))
    print("{} 중간 값: {}".format(name, np.median(counts)))
    # 4분위의 경우 0~100 스케일로 되어 있음
    print("{} 제 1 사분위: {}".format(name, np.percentile(counts, 25)))
    print("{} 제 3 사분위: {}".format(name, np.percentile(counts, 75)))


def print_char_length_per_review(train_data):
    # 각 리뷰의 문자 길이 분포 산출
    train_length = train_data["review"].apply(len)
    train_length.head()
    plt.figure(figsize=(12, 5))
    plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
    plt.yscale('log')
    plt.title('Log-Histogram of length of review')
    plt.xlabel('Length of review')
    plt.ylabel('Number of review')
    plt.show()
    # 리뷰 길이 통계 산출
    print_statistics("리뷰 길이", train_length)
    plt.figure(figsize=(12, 5))
    plt.boxplot(train_length, labels=["counts"], showmeans=True)


def visualize_words(train_data):
    # 가장 많이 나온 단어에 대한 시각화 표출
    cloud = WordCloud(width=800, height=600).generate("".join(train_data["review"]))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis("off")


def show_pos_and_neg_propotion(train_data):
    # 긍정 및 부정 리뷰의 개수 및 분포 그래프 표출
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_data["sentiment"])
    plt.show()
    print("긍정 리뷰 개수: {}".format(train_data["sentiment"].value_counts()[1]))
    print("부정 리뷰 개수: {}".format(train_data["sentiment"].value_counts()[0]))


def show_word_cnt_per_review(train_data):
    # 리뷰 당 단어 갯수 통계 및 그래프 표출
    train_word_cnts = train_data["review"].apply(lambda x: len(x.split(" ")))
    plt.figure(figsize=(15, 10))
    plt.hist(train_word_cnts, bins=50, facecolor='r', label='train')
    plt.title("Log-Histogram of word count in review", fontsize=15)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Number of words", fontsize=15)
    plt.ylabel("Number of reviews", fontsize=15)
    plt.show()
    # 단어 개수 통계 산출
    print_statistics("단어 개수", train_word_cnts)


def show_propotion_of_marks(train_data):
    # 특수문자 및 대소문자 비율 산출
    qmarks = np.mean(train_data["review"].apply(lambda x: '?' in x))
    fullstop = np.mean(train_data["review"].apply(lambda x: '.' in x))
    capital_first = np.mean(train_data["review"].apply(lambda x: x[0].isupper()))
    capital_cnt = np.mean(train_data["review"].apply(lambda x: max([y.isupper() for y in x])))
    number_cnt = np.mean(train_data["review"].apply(lambda x: max([y.isdigit() for y in x])))
    print("물음표가 있는 질문: {:2f}%".format(qmarks * 100))
    print("마침표가 있는 질문: {:2f}%".format(fullstop * 100))
    print("첫 글자가 대문자인 질문: {:2f}%".format(capital_first * 100))
    print("대문자가 있는 질문: {:2f}%".format(capital_cnt * 100))
    print("숫자가 있는 질문: {:2f}%".format(number_cnt * 100))


def data_preprocess(train_data):
    # 데이터 전처리 수행 (전처리 되기 전 데이터 예제 하나 표출)
    print(train_data["review"][0])
    # HTML 태그 제거 작업
    review = train_data["review"][0]
    review_text = BeautifulSoup(review, "html5lib").get_text()
    # 문장 부호 및 영어가 아닌 문자 제거 작업
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 영어 불용어 처리
    stop_words = set(stopwords.words("english"))
    review_text = review_text.lower()
    words = review_text.split()
    words = [w for w in words if w not in stop_words]
    print(words)
    # 불용어 필터링 된 단어 리스트를 다시 줄글로 합침
    clean_review = " ".join(words)
    print(clean_review)


def main():
    # 메인 함수
    print("시작")
    # 데이터를 로딩
    dataset = load()
    # 데이터 확인 및 갯수 세리기
    print_data_size(dataset)
    # 각 리뷰의 문자 길이 분포 그래프 표출
    print_char_length_per_review(dataset)
    # 많이 사용된 단어 시각화
    visualize_words(dataset)
    # 긍/부정 데이터 분포 그래프 표출
    show_pos_and_neg_propotion(dataset)
    # 각 리뷰의 단어 개수 분포 그래프 표출
    show_word_cnt_per_review(dataset)
    # 특수문자 및 대소문자 비율 산출
    show_propotion_of_marks(dataset)
    # 첫 번째 리뷰 데이터에 대한 데이터 전처리 수행
    data_preprocess(dataset)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
