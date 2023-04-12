import numpy as np
import pandas as pd
import os
import re
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Okt
from wordcloud import WordCloud
from tqdm import tqdm
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping

from week6.CNNClassifier import CNNClassifier

matplotlib.use('TkAgg')

# JVM 경로
JVM_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home/jre/lib/server/libjvm.dylib"

# 데이터 입출력 경로
DATA_IN_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week6/data/kr/"
DATA_OUT_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week6/data/kr_out/"


def load_data():
    print('파일 크기 : ')
    # 경로에 있는 txt 파일을 로드하고 크기를 메가바이트 단위로 소수점 2째 자리에서 반올림하여 파일별 용량 확인
    for file in os.listdir(DATA_IN_PATH):
        if 'txt' in file:
            print(file.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + 'MB')

    train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header=0, delimiter='\t', quoting=3)
    train_data.head()
    return train_data


def load_test_data():
    test_data = pd.read_csv(DATA_IN_PATH + 'ratings_test.txt', header=0, delimiter='\t', quoting=3)
    return test_data


def analyse_data(train_data):
    # 데이터 분석 함수
    # 데이터의 개수를 확인
    print('전체 학습데이터의 개수: {}'.format(len(train_data)))
    train_length = train_data['document'].astype(str).apply(len)
    train_length.head()

    # 히스토그램 선언
    # bins: 히스토그램 값들에 대한 버켓 범위
    # range: x축 범위
    # alpha: 그래프 투명도
    # color: 그래프 색상
    # label: 그래프에 대한 라벨
    plt.figure(figsize=(12, 5))
    plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
    plt.yscale('log')
    # 그래프 제목
    plt.title('Log-Histogram of length of review')
    # 그래프 x축 라벨
    plt.xlabel('Length of review')
    # 그래프 y축 라벨
    plt.ylabel('Number of review')
    plt.show()

    # 리뷰 통계정보 살펴보기
    print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
    print('리뷰 길이 최소값: {}'.format(np.min(train_length)))
    print('리뷰 길이 평균값: {:.2f}'.format(np.mean(train_length)))
    print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
    print('리뷰 길이 중앙값: {}'.format(np.median(train_length)))
    print('리뷰 길이 제 1 사분위값 (25%): {}'.format(np.percentile(train_length, 25)))
    print('리뷰 길이 제 3 사분위값 (75%): {}'.format(np.percentile(train_length, 75)))

    # 박스플롯을 통해 리뷰 분포 살펴보기
    # boxplot의 첫번째 파라미터: 분포에 대한 데이터 리스트
    # labels: 입력한 데이터에 대한 라벨
    # showmeans: 평균값
    plt.figure(figsize=(12, 5))
    plt.boxplot(train_length, labels=['counts'], showmeans=True)
    plt.show()

    # 워드 클라우드 이미지 형태로 자주 사용되는 단어 보기
    train_review = [review for review in train_data['document'] if type(review) is str]
    wordcloud = WordCloud(font_path=DATA_IN_PATH + 'NanumGothic.ttf').generate(''.join(train_review))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # 긍정, 부정 리뷰의 개수 살펴보기
    fig, axe = plt.subplot(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_data['label'])
    plt.show()
    print('긍정 리뷰 개수: {}'.format(train_data['label'].value_counts()[1]))
    print('부정 리뷰 개수: {}'.format(train_data['label'].value_counts()[0]))

    # 리뷰 단어 개수 살펴보기
    train_word_counts = train_data['document'].astype(str).apply(lambda x: len(x.split(' ')))
    plt.figure(figsize=(15, 10))
    plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
    plt.title('Log-Histogram of word count in review', fontsize=15)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Number of reviews', fontsize=15)
    plt.show()

    # 리뷰 단어 개수 통계정보 살펴보기
    # 리뷰 통계정보 살펴보기
    print('리뷰 단어 개수 최댓값: {}'.format(np.max(train_word_counts)))
    print('리뷰 단어 개수 최소값: {}'.format(np.min(train_word_counts)))
    print('리뷰 단어 개수 평균값: {:.2f}'.format(np.mean(train_word_counts)))
    print('리뷰 단어 개수 표준편차: {:.2f}'.format(np.std(train_word_counts)))
    print('리뷰 단어 개수 중앙값: {}'.format(np.median(train_word_counts)))
    print('리뷰 단어 개수 제 1 사분위값 (25%): {}'.format(np.percentile(train_word_counts, 25)))
    print('리뷰 단어 개수 제 3 사분위값 (75%): {}'.format(np.percentile(train_word_counts, 75)))

    # 물음표 및 마침표가 있는 질문 찾기
    qmarks = np.mean(train_data['document'].astype(str).apply(lambda x: '?' in x))
    fullstop = np.mean(train_data['document'].astype(str).apply(lambda x: '.' in x))
    print('물음표가 있는 질문: {:.2f}%'.format(qmarks * 100))
    print('마침표가 있는 질문: {:.2f}%'.format(fullstop * 100))


def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    # 각 리뷰 별 전처리를 수행하는 함수
    # 함수의 인자는 다음과 같다
    # review = 전처리할 텍스트
    # okt = 재사용할 okt 토크나이저 객체
    # remove_stopword: 불용어 제거 여부
    # stop_word : 불용어 사전
    review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', review)

    word_review = okt.morphs(review_text, stem=True)

    if remove_stopwords:
        word_review = [token for token in word_review if token not in stop_words]
    return word_review


def preprocess_data(train_data, test_data):
    # 전체 데이터에 대한 전처리 수행 함수
    # 형태소 분석 및 어근 추출
    stop_words = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
    okt = Okt(jvmpath=JVM_HOME)
    clean_train_review = []
    clean_test_review = []

    # 학습 데이터 형태소 분석
    for review in tqdm(train_data['document']):
        # 코드 실행 시간이 긺으로 tqdm으로 볼 수 있게 함
        # 비어있는 데이터에서 멈추지 않도록 데이터타입이 string인 경우만 필터링
        if type(review) == str:
            clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
        else:
            # 스트링이 아닌 경우에 대한 예외 처리
            clean_train_review.append([])

    # 테스트 데이터 형태소 분석
    for review in tqdm(test_data['document']):
        if type(review) == str:
            clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
        else:
            # 스트링이 아닌 경우에 대한 예외 처리
            clean_test_review.append([])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_train_review)
    train_sequences = tokenizer.texts_to_sequences(clean_train_review)
    test_sequences = tokenizer.texts_to_sequences(clean_test_review)

    # 단어 사전 생성
    word_vocab = tokenizer.word_index
    word_vocab['<PAD>'] = 0

    # 문장 길이 정리 및 라벨 데이터와 매핑
    max_sequence_length = 8
    train_inputs = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')
    train_labels = np.array(train_data['label'])
    test_inputs = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')
    test_labels = np.array(test_data['label'])

    # 데이터 설정
    data_configs = {
        'vocab': word_vocab,
        'vocab_size': len(word_vocab)
    }

    # 파일에 쓰기
    np.save(open(DATA_IN_PATH + 'nsmc_train_input.npy', 'wb'), train_inputs)
    np.save(open(DATA_IN_PATH + 'nsmc_train_label.npy', 'wb'), train_labels)
    np.save(open(DATA_IN_PATH + 'nsmc_test_input.npy', 'wb'), test_inputs)
    np.save(open(DATA_IN_PATH + 'nsmc_test_label.npy', 'wb'), test_labels)
    json.dump(data_configs, open(DATA_IN_PATH + 'data_configs.json', 'w'), ensure_ascii=False)


def plot_graphs(history, string):
    # 모델 시각화 함수
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epoches')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def train_model():
    seed_num = 1234
    tf.random.set_seed(seed_num)

    train_input = np.load(open(DATA_IN_PATH + 'nsmc_train_input.npy', 'rb'))
    train_label = np.load(open(DATA_IN_PATH + 'nsmc_train_label.npy', 'rb'))
    prepro_configs = json.load(open(DATA_IN_PATH + 'data_configs.json', 'r'))

    model_name = 'cnn_classifier_kr'
    batch_size = 512
    num_epoches = 10
    valid_split = 0.1
    max_len = train_input.shape[1]

    kargs = {
        'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
        'embedding_size': 128,
        'num_filters': 100,
        'dropout_rate': 0.5,
        'hidden_dimension': 250,
        'output_dimension': 1
    }
    # noinspection DuplicatedCode
    model = CNNClassifier(**kargs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

    # 콜백 선언 - 오버피팅을 막기 위한 earlystop 추가
    # 0.0001 이상의 정확도 상승이 2회 이상 없으면 중지
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)

    # noinspection DuplicatedCode
    checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_dir):
        print('{} -- Folder already exists \n'.format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print('{} -- Folder create complete \n'.format(checkpoint_dir))

    cp_callback = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True
    )

    # 모델 학습
    history = model.fit(
        train_input, train_label, batch_size=batch_size, epochs=num_epoches, validation_split=valid_split,
        callbacks=[earlystop_callback, cp_callback]
    )

    plot_graphs(history, 'loss')
    plot_graphs(history, 'accuracy')

    return {
        'model': model,
        'history': history,
        'batch_size': batch_size
    }


def validate(model):
    test_input = np.load(open(DATA_IN_PATH + 'nsmc_test_input.npy', 'rb'))
    test_input = pad_sequences(test_input, maxlen=test_input.shape[1])
    test_label = np.load(open(DATA_IN_PATH + 'nsmc_test_label.npy', 'rb'))

    model.load_weights(os.path.join(DATA_OUT_PATH, 'cnn_classifier_kr', 'weights.h5'))
    model.evaluate(test_input, test_label)


train = load_data()
test = load_test_data()
preprocess_data(train, test)
result = train_model()
validate(result['model'])
