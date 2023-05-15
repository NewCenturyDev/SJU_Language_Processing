from week11_2 import *
import json

DATA_IN_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week11/data/Chatbot_data.csv"
DATA_IN_FILE_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week11/data/Chatbot_data.csv"
VOCAB_PATH = "/Users/newcentury99/Documents/SJU_Language_Processing/week11/data/vocabulary.txt"

inputs, outputs = load_data(DATA_IN_FILE_PATH)
word2idx, idx2word, vocab_size = load_vocabulary(DATA_IN_FILE_PATH, VOCAB_PATH, tokenize_as_morph=False)

index_inputs, input_seq_len = enc_processing(inputs, word2idx, tokenize_as_morph=False)
index_outputs, output_seq_len = dec_output_processing(outputs, word2idx, tokenize_as_morph=False)
index_targets = dec_target_processing(outputs, word2idx, tokenize_as_morph=False)

data_confings = {
    'word2idx': word2idx,
    'idx2word': idx2word,
    'vocab_size': vocab_size,
    'pad_symbol': PAD,
    'std_symbol': STD,
    'end_symbol': END,
    'unk_symbol': UNK
}

TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'tarin_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH + TRAIN_INPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH + TRAIN_OUTPUTS, 'wb'), index_outputs)
np.save(open(DATA_IN_PATH + TRAIN_TARGETS, 'wb'), index_targets)

json.dump(data_confings, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))
