# 2주차-1: 텍스트 유사도를 구해 보기

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import numpy


def vectorise_text(sentenses):
    vectorizer = TfidfVectorizer()
    # 문장 벡터화 진행
    tfidf_matrix = vectorizer.fit_transform(sentenses)

    idf = vectorizer.idf_
    print("텍스트를 벡터라이징 하여 idf(특정 단어가 있는 문장의 개수 / 전체 문장의 개수) 크기 확인")
    print(dict(zip(vectorizer.get_feature_names_out(), idf)))
    return tfidf_matrix


def l1_normalize(vector):
    # L1 노말라이징 진행
    norm = numpy.sum(vector)
    return vector / norm


def calc_cosine_simularity(tfidf_matrix):
    # 코사인 유사도 측정
    print("코사인 유사도를 측정하여 두 문장간의 유사도를 산출")
    simularity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print(simularity)


def calc_euclidean_simularity(tfidf_matrix):
    # L1 정규화 진행
    tfidf_norm_l1 = l1_normalize(tfidf_matrix)

    # 유클리디안 유사도 측정
    print("유클리디안 유사도를 측정하여 두 문장간의 유사도를 산출")
    distance = euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
    print(distance)


def calc_manhattan_simularity(tfidf_matrix):
    # L1 정규화 진행
    tfidf_norm_l1 = l1_normalize(tfidf_matrix)

    # 맨해튼 유사도 측정
    print("맨해튼 유사도를 측정하여 두 문장간의 유사도를 산출")
    distance = manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
    print(distance)


def main():
    # 메인 함수
    print("시작")
    sentenses = (
        "휴일 인 오늘 도 서쪽 을 중심 으로 폭염 이 이어졌는데요, 내일 은 반가운 비 소식 이 있습니다.",
        "폭염 을 피해서 휴일 에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니 다."
    )
    tfidf_matrix = vectorise_text(sentenses)
    calc_cosine_simularity(tfidf_matrix)
    calc_euclidean_simularity(tfidf_matrix)
    calc_manhattan_simularity(tfidf_matrix)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
