import tensorflow as tf
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from week11_2 import *

# seq2seq.py


DATA_IN_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week11/data/"
DATA_OUT_PATH = '/Users/newcentury99/Documents/SJU_Language_Processing/week11/result/'
TRAIN_INPUT = "train_inputs.npy"
TRAIN_OUTPUT = "train_outputs.npy"
TRAIN_TARGETS = "train_targets.npy"
DATA_CONFIGS = "data_configs.json"

SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string], '')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# 전처리된 데이터 로드
index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUT, 'rb'))[:100]
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUT, 'rb'))[:100]
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS, 'rb'))[:100]
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
print(len(index_inputs), len(index_outputs), len(index_targets))

# 모델 구성에 필요한 값 선언
MODEL_NAME = 'seq2seq_kor'
BATCH_SIZE = 2
MAX_SEQUENCE = 25
EPOCH = 30
UNITS = 1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1

char2idx = prepro_configs['word2idx']
idx2char = prepro_configs['idx2word']
std_index = prepro_configs['std_symbol']
end_index = prepro_configs['end_symbol']
vocab_size = prepro_configs['vocab_size']


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,  # 각 시퀀스마다 출력을 반환할지 여부 결정
            return_state=True,  # 마지막 상태값을 반환
            recurrent_initializer='glorot_uniform'  # GRU 신경망을 만드는 부분
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)  # 임베딩 벡터와 RNN 초기화 상태로 은닉 상태 전달
        return output, state  # 시퀀스의 출력값과 마지막 상태값 리턴

    def initialize_hidden_state(self, inp):  # 배치 크기를 받아 RNN 초기에 사용될 크기의 은닉 상태를 만듦
        return tf.zeros((tf.shape(inp)[0], self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):  # 인코더 은닉층의 상태값, 결과값
        hidden_with_time_axis = tf.expand_dims(query, 1)  # 행렬곱을 할 수 있게 변환

        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()

        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        self.attention = BahdanauAttention(self.dec_units)  # 어텐션이 계산된 문맥 벡터를 돌려받음

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)
        return x, state, attention_weights


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='accuracy')


def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def accuarcy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask
    acc = train_accuracy(real, pred)

    return tf.reduce_mean(acc)


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_sz, end_token_idx=2):
        super(Seq2Seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_sz)

    def call(self, x):
        inp, tar = x

        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        predict_tokens = list()
        for t in range(0, tar.shape[1]):  # 시퀀스의 길이만큼 반복하면서 디코더의 출력값을 만들어냄
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32)
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))
        return tf.stack(predict_tokens, axis=1)

    def interfence(self, x):
        inp = x
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([char2idx[std_index]], 1)

        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])
            if predict_token == self.end_token_idx:
                break

            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)

        return tf.stack(predict_tokens, axis=0).numpy()


model = Seq2Seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE, char2idx[end_index])
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[accuarcy])

PATH = DATA_OUT_PATH + MODEL_NAME
if not(os.path.isdir(PATH)):
    os.makedirs(os.path.join(PATH))

checkpoint_path = DATA_OUT_PATH + MODEL_NAME + '/weights.h5'
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True
)

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)

history = model.fit([index_inputs, index_outputs], index_targets,
                    batch_size=BATCH_SIZE, epochs=EPOCH,
                    validation_split=VALIDATION_SPLIT, callbacks=[earlystop_callback, cp_callback])

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

SAVE_FILE_NM = 'weights.h5'
model.load_weights(os.path.join(DATA_OUT_PATH, MODEL_NAME, SAVE_FILE_NM))

query = '남자친구 승진 선물로 뭐가 좋을까?'

test_index_inputs, _ = enc_processing([query], char2idx)
predict_tokens = model.interfence(test_index_inputs)
print(predict_tokens)
print(''.join([idx2char[str(t)] for t in predict_tokens]))

