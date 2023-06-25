import json
import os
import platform
import re

import nltk
from keras.layers import Dense, Dropout, Embedding, LSTM, Convolution1D, MaxPooling1D, Activation
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords

# 환경변수
NLTK_PATH_WINDOWS = '/project/nltk_data'
NLTK_PATH_LINUX = '/app/data'
nltk.data.path.append(NLTK_PATH_WINDOWS if platform.system() == 'Windows' else NLTK_PATH_LINUX)

PATH = './result/nlp_classifier_en/'
TOKENIZER_PATH = './preprocessed/tokenizer.json'
SAVE_FILE_NM = 'weights.h5'  # 저장된 best model 이름
MAX_SEQ_LEN = 52


def load_pretrained_model():
    # 모델 불러오기
    model_name = 'nlp_classifier_en'
    kargs = {
        'model_name': model_name,
        'vocab_size': 113677,
        'embedding_dimension': 32,
        'dropout_rate': 0.5,
        'cnn_dimension': 128,
        'lstm_dimension': 128,
        'dense_dimension': 128,
        'output_dimension': 3
    }
    # noinspection DuplicatedCode
    model = Sequential()
    model.add(Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_dimension']))
    model.add(Convolution1D(kargs['cnn_dimension'], kargs['output_dimension'], padding='same', strides=1))
    model.add(Activation('leaky_relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(kargs['lstm_dimension'], return_sequences=True, activation='leaky_relu'))
    model.add(LSTM(kargs['lstm_dimension'], activation='leaky_relu'))
    model.add(Dropout(kargs['dropout_rate']))
    model.add(Dense(kargs['dense_dimension'], activation='leaky_relu'))
    model.add(Dropout(kargs['dropout_rate']))
    model.add(Dense(kargs['output_dimension'], activation='softmax'))
    model.build(input_shape=(kargs['vocab_size'], kargs['embedding_dimension']))
    model.load_weights(os.path.join(PATH, SAVE_FILE_NM))
    return model


def do_remove_stopwords(words):
    # 영어 불용어 처리
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return words


def preprocessing(text, remove_stopwords=False):
    # 비 알파벳 문자 제거
    text_text = re.sub("[^a-zA-Z]", " ", text)

    # 대문자를 전부 소문자로 변환하고 공백 단위로 토크나이징
    words = text_text.lower().split()

    # 불용어 제거
    if remove_stopwords:
        words = do_remove_stopwords(words)
    # 불용어 필터링 처리 또는 처리하지 않은 단어 리스트를 다시 줄글로 합침
    clean_text = " ".join(words)
    return clean_text


def prep_test_data(sentence):
    clean_text = preprocessing(sentence, True)
    # 저장된 토크나이저 로드
    saved_tokenizer = tokenizer_from_json(json.load(open(TOKENIZER_PATH, 'r', encoding='utf-8')))

    # 토크나이징 객체를 새롭게 만들면 인덱스가 바뀌어 적절하게 평가하거나 테스트 할 수 없으므로 최초 생성된 객체 사용
    print([clean_text])
    text_sequences = saved_tokenizer.texts_to_sequences([clean_text])
    print(text_sequences)
    vectorized_test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQ_LEN, padding="post")

    # 벡터화된 테스트 데이터 반환
    return vectorized_test_inputs


def do_validation(test_input):
    test_model = load_pretrained_model()
    test_input_vector = prep_test_data(test_input)
    
    # 예측 진행
    batch_size = 512
    predictions = test_model.predict(test_input_vector, batch_size=batch_size)
    decode_map = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    category = decode_map[predictions[0].argmax()]
    return category


# noinspection DuplicatedCode
def main():
    test_sentence = "shit, I messed my exam."
    test_predicted = do_validation(test_sentence)
    print(test_predicted)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조
