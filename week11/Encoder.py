
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz # 배치 크기
        self.enc_units = enc_units # 재귀 신경망의 결과 차원
        self.vocab_size = vocab_size # 사전 크기
        self.embedding_dim = embedding_dim # 임베딩 차원
        # 사전에 포함된 각 단어를 embedding_dim 차원의 임베딩 벡터로 만듦
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        # 임베딩 벡터와 RNN 초기화 상태로 은닉 상태 전달
        output, state = self.gru(x, initial_state=hidden)
        # 시퀀스의 출력값과 마지막 상태값 리턴
        return output, state

    # 배치 크기를 받아 RNN 초기에 사용될 크기의 은닉 상태를 만듦
    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0], self.enc_units))
