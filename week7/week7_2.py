import pandas as pd
import numpy as np
import xgboost as xgb
import os

from sklearn.model_selection import train_test_split

DATA_IN_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/'
PREPED_DATA_IN_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/preped/'
DATA_OUT_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/result/'


def load_preped_train_data():
    train_q1_data = np.load(open(PREPED_DATA_IN_PATH + 'train_q1.npy', 'rb'))
    train_q2_data = np.load(open(PREPED_DATA_IN_PATH + 'train_q2.npy', 'rb'))
    train_labels = np.load(open(PREPED_DATA_IN_PATH + 'train_label.npy', 'rb'))
    return {
        'train_q1': train_q1_data,
        'train_q2': train_q2_data,
        'label': train_labels
    }


def load_preped_test_data():
    test_q1_data = np.load(open(PREPED_DATA_IN_PATH + 'test_q1.npy', 'rb'))
    test_q2_data = np.load(open(PREPED_DATA_IN_PATH + 'test_q2.npy', 'rb'))
    test_id_data = np.load(open(PREPED_DATA_IN_PATH + 'test_id.npy', 'rb'))
    return {
        'test_q1': test_q1_data,
        'test_q2': test_q2_data,
        'id': test_id_data
    }


def do_train(train_q1_data, train_q2_data, train_labels):
    train_input = np.stack((train_q1_data, train_q2_data), axis=1)
    print(train_input.shape)

    # 학습 데이터에서 검증 데이터 분리
    train_input, eval_input, train_label, eval_label = train_test_split(
        train_input, train_labels, test_size=0.2, random_state=4242
    )

    # 데이터를 DMatrix로 읽어오기
    train_data = xgb.DMatrix(train_input.sum(axis=1), label=train_label)
    eval_data = xgb.DMatrix(eval_input.sum(axis=1), label=eval_label)
    data_list = [(train_data, 'train'), (eval_data, 'valid')]

    # 인자를 설정 (로지스틱 예측 사용, root_mean_square_error - 평균 제곱 오차 사용)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
    }

    # 10회 연속 변화 없을 시 중지, 최대 1000회 반복시 중지, XG 부스트 모델 알고리즘을 사용하여 학습 진행
    bst = xgb.train(params, train_data, num_boost_round=1000, evals=data_list, early_stopping_rounds=10)
    return bst


def do_test(test_q1_data, test_q2_data, test_id_data, bst):
    test_input = np.stack((test_q1_data, test_q2_data), axis=1)
    test_data = xgb.DMatrix(test_input.sum(axis=1))
    test_predict = bst.predict(test_data)

    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    output = pd.DataFrame({'test_id': test_id_data, 'is_duplicate': test_predict})
    output.to_csv(DATA_OUT_PATH + 'simple_xgb.csv', index=False)


train_dataset = load_preped_train_data()
xg_boost_model = do_train(train_dataset['train_q1'], train_dataset['train_q2'], train_dataset['label'])
test_dataset = load_preped_test_data()
do_test(test_dataset['test_q1'], test_dataset['test_q2'], test_dataset['id'], xg_boost_model)
