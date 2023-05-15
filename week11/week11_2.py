
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

# preprocess.py

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"  # 의미 없는 패딩 토큰
STD = "<SOS>"  # 시작 토큰을 의미
END = "<END>"  # 종료 토큰을 의미
UNK = "<UNK>"  # 사전에 없는 단어를 의미

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEK = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)  # 정규표현식 모듈을 사용해 컴파일

MAX_SEQUENCE = 25


def load_data(path):
    # 판다스를 통해서 데이터를 불러온다
    data_df = pd.read_csv(path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 얺는다
    question, answer = list(data_df['Q']), list(data_df['A'])
    return question, answer


# 정규표현식을 이용해 특수 기호를 모두 제거하고, 공백 문자를 기준으로 단어를 나눠서 리스트로 만드는 함수
def data_tokenizer(data):
    # 토크나이징 된 단어를 담을 배열
    words = []
    # FILTER와 같은 값을 정규표현식을 통해 제거
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, '', sentence)
        for word in sentence.split():
            words.append(word)
    # 토크나이징과 정규표현식을 통해 만들어진 값들을 반환
    return [word for word in words if word]


# 한글 텍스트를 토크나이징 하기 위해 형태소로 분리하는 함수
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = ' '.join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data


# 사전 내에 단어 하나를 추가하는 함수
def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리르 만든다
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스이고 값이 단어인 딕셔너리로 만든다
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    # 두개의 딕셔너리를 넘겨 준다
    return word2idx, idx2word


# 사전을 만드는 함수
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    # 사전을 담을 배열 준비한다
    vocabulary_list = []
    # 사전을 구성한 후 파일로 저장 진행 및 파일 존재 유무 확인
    if not os.path.exists(vocab_path):
        if os.path.exists(path):
            # 데이터가 존재하면 판다스를 통해 데이터 로드
            data_df = pd.read_csv(path, encoding='utf-8')
            # 데이터프레임을 통해 질문과 답에 대한 열을 가져온다
            question, answer = list(data_df['Q']), list(data_df['A'])
            # 형태소에 따른 토크나이저 처리
            if tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            data = []
            # 질문과 답변을 extend를 통해서 구조가 없는 배열로 만든다
            data.extend(question)
            data.extend(answer)
            # 토크나이저 처리 하는 부분이다
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER
            # 사전을 리스트로 만들었으니 이 내용을 사전 파일로 만들어 넣는다
            with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
                for word in words:
                    vocabulary_file.write(word + '\n')

            # 사전 파일이 존재하면 여기에서 그 파일을 불러서 배열에 넣어 준다
            with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
                for line in vocabulary_file:
                    vocabulary_list.append(line.strip())

            # 배열에 내용을 키와 값이 있는 딕셔너리 구조로 만든다
            word2idx, idx2word = make_vocabulary(vocabulary_list)
            # 두가지 형태의 키와 값이 있는 형태를 리턴한다
            return word2idx, idx2word, len(word2idx)


# 인코더에 적용될 입력값을 만드는 전처리 함수
def enc_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는 배열이다. (누적된다.)
    sequences_input_index = []
    # 하나의 인코딩 되는 문장의 길이를 가지고 있다 (누적된다)
    sequence_length = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    # 한줄씩 불러온다
    for sequence in value:
        # 정규화로 필터에 있는 특수기호 제거
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        # 하나의 문장을 인코딩 할 때 가지고 있기 위한 배열
        sequence_index = []
        # 문장을 스페이스 단위로 자르고 있다

        for word in sequence.split():
            # 잘려진 단어들이 딕셔너리에 존재하는지 보고 그 값을 가져와 sequence_index에 추가한다
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                # 딕셔너리에 없는 단어임으로 UNK를 넣어 준다
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequence_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequence_length


# 디코더에 적용될 입력값을 만드는 전처리 함수
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가진 배열
    sequences_output_index = []
    # 하나의 디코딩 입력 되는 문장의 길이
    sequences_length = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한 줄씩 불러온다
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        # sequence_index = []
        sequence_index = [dictionary[STD]] + \
                         [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        # 문장 제한 길이보다 길어질 경우 뒤의 토큰을 자르고 END 토큰을 넣어준다
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        # 하나의 문장에 길이를 넣어주고 있다
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다
    # 이유는 텐서플로우 dataset에 넣어 주기 위한 사정 작업이다
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다
    return np.asarray(sequences_output_index), sequences_length


# 디코더에 적용될 타겟값을 만드는 전처리 함수
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가진 배열
    sequences_target_index = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한 줄씩 불러온다
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        # sequence_index = []
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        # 문장 제한 길이보다 길어질 경우 뒤의 토큰을 자르고 END 토큰을 넣어준다
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            # 짧은 경우 PAD를 채워준다
            sequence_index += [dictionary[END]]
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_target_index에 넣어 준다
        sequences_target_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다
    # 이유는 텐서플로우 dataset에 넣어 주기 위한 사정 작업이다
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다
    return np.asarray(sequences_target_index)
