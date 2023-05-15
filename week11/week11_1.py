import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from konlpy.tag import Okt

from functools import reduce
from wordcloud import WordCloud

DATA_IN_PATH = './data'
data = pd.read_csv(DATA_IN_PATH + 'ChatBotData.csv', encoding='utf-8')

print("===== 데이터프레임 출력 =====")
print(data)

# 질문과 답변 모두에 대해 길이를 분석
sentences = list(data['Q']) + list(data['A'])

# 띄어쓰기 기준으로 문장을 나눔
tokenized_sentences = [s.split0 for s in sentences]
# 이 값을 이용해 어절 길이 측정
sent_len_by_token = [len(t)for t in tokenized_sentences]
# 공백을 제거 후 음절단위 길이 계산
sent_len_by_eumjeol = [len(s.replace(' ', '')) for s in sentences]
# 형태소 분석기를 사용해서 나눈 후 길이를 측정
okt = Okt()
morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in sentences]
sent_len_by_morph = [len(t) for t in morph_tokenized_sentences]


