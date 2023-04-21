import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from SentenceMaLSTM import MaLSTM

matplotlib.use('TkAgg')
PREPED_DATA_IN_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/preped/'
DATA_OUT_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week7/data/result/'


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def load_train_data():
    q1_data = np.load(open(PREPED_DATA_IN_PATH + 'train_q1.npy', 'rb'))
    q2_data = np.load(open(PREPED_DATA_IN_PATH + 'train_q2.npy', 'rb'))
    labels = np.load(open(PREPED_DATA_IN_PATH + 'train_label.npy', 'rb'))
    prepro_configs = json.load(open(PREPED_DATA_IN_PATH + 'data_configs.json', 'r'))
    return {
        'q1': q1_data,
        'q2': q2_data,
        'labels': labels,
        'configs': prepro_configs
    }


def load_test_data():
    q1_data = np.load(open(PREPED_DATA_IN_PATH + 'test_q1.npy', 'rb'))
    q2_data = np.load(open(PREPED_DATA_IN_PATH + 'test_q2.npy', 'rb'))
    ids = np.load(open(PREPED_DATA_IN_PATH + 'test_id.npy', 'rb'))
    return {
        'q1': q1_data,
        'q2': q2_data,
        'id': ids,
    }


def define_model_hyperparams(prepro_configs):
    seed_num = 1234
    tf.random.set_seed(seed_num)

    model_name = 'malstm_similarity'
    max_len = 31

    kargs = {
        'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
        'embedding_dimension': 100,
        'lstm_dimension': 150,
    }

    model = MaLSTM(**kargs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
    return model


def do_train(model, q1_data, q2_data, labels):
    batch_size = 128
    num_epoches = 5
    valid_split = 0.1

    # overfitting을 막기 위한 ealrystop 추가j
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=1)
    # min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
    # patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

    checkpoint_path = DATA_OUT_PATH + 'malstm_similarity' + '/weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create path if exists
    if os.path.exists(checkpoint_dir):
        print("{} -- Folder already exists \n".format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("{} -- Folder create complete \n".format(checkpoint_dir))

    cp_callback = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    history = model.fit((q1_data, q2_data), labels, batch_size=batch_size, epochs=num_epoches,
                        validation_split=valid_split, callbacks=[earlystop_callback, cp_callback])

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


def do_test(model, test_q1_data, test_q2_data, test_id_data):
    batch_size = 128
    save_file_name = 'weights.h5'
    model.load_weights(os.path.join(DATA_OUT_PATH, 'malstm_similarity', save_file_name))

    predictions = model.predict((test_q1_data, test_q2_data), batch_size=batch_size)
    predictions = predictions.squeeze(-1)

    output = pd.DataFrame(data={'test_id': test_id_data, 'is_duplicate': list(predictions)})
    output.to_csv('malstm_similarity.csv', index=False, quoting=3)


train_data = load_train_data()
lstm_model = define_model_hyperparams(train_data['configs'])
do_train(lstm_model, train_data['q1'], train_data['q2'], train_data['labels'])
test_data = load_test_data()
do_test(lstm_model, test_data['q1'], test_data['q2'], test_data['id'])
