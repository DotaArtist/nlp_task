#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bilstm 文本分类"""

__author__ = 'yp'

from tensorflow.python import debug as tf_debug
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import collections
import random
import numpy as np
from pyltp import Segmentor
import random
import numpy as np
import pickle
import math
import re
from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell
from operator import mul
from functools import reduce

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def attention(inputs, attention_size, name="att", train_type=True):
    """
    Attention mechanism layer.

    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :param name:name of op
    :param train_type:for test, non random uninitialized
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    with tf.name_scope(name):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)

        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        if train_type:
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='att_w')

            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_b')

            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_u')

            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + tf.reshape(b_omega, [1, -1]), name='att_v')
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]), name='att_vu')
            exps = tf.reshape(tf.exp(vu), [-1, sequence_length], name='att_exps')
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1], name='att_alphas')

            # Output of Bi-RNN is reduced with attention vector
            output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1, name='att_output')
        else:
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='att_w')

            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_b')

            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_u')

            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + tf.reshape(b_omega, [1, -1]), name='att_v')
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]), name='att_vu')
            exps = tf.reshape(tf.exp(vu), [-1, sequence_length], name='att_exps')
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1], name='att_alphas')

            # Output of Bi-RNN is reduced with attention vector
            output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1, name='att_output')

        return output, w_omega


# CWS_MODEL_PATH = "C:/Users/YP_TR/Desktop/emotion/data/ltp_data/cws.model"


def copy(listx, label, times):
    for i in range(times):
        listx.append(label)


def copy_ex(listx, label, times):
    for i in range(times):
        listx.extend(label)


def load_train(path_list, dictionary):
    """
    :param path:
    :param min_cout:
    :return:<class 'list'>: [[487, 12, 1042, 1, 262, 121, 1991, 20], 0]
    <class 'dict'>: {'记得': 1889, '吗##我': 609, '下地': 3065, '认': 1942, '外壳': 1765, '尊锐': 1274}
    """

    datas_word = []
    labels = []
    words = []

    # seg model
    # segmentor = Segmentor()
    # segmentor.load(CWS_MODEL_PATH)

    for path in path_list:
        reader = open(path, 'r', encoding='utf-8')

        for line in reader.readlines():

            try:
                _, label, sentences = line.strip().split('\t')
                # sentences = " ".join(segmentor.segment(sentences))
            except:
                print("train line error:", line)
                continue

            sentences = sentences.split()

            # no mapping
            # labels.append(label)
            # words.extend(sentences)

            # label mapping and sampling
            if label in ['0', '4']:
                if random.randint(0, 0) == 0:
                    copy(labels, 1, 1)
                    copy(datas_word, sentences, 1)
                    copy_ex(words, sentences, 1)
                else:
                    continue
            elif label in ['1', '3']:
                copy(labels, 2, 1)
                copy(datas_word, sentences, 1)
                copy_ex(words, sentences, 1)
            elif label in ['2', '5', '6', '7']:
                copy(labels, 0, 2)
                copy(datas_word, sentences, 2)
                copy_ex(words, sentences, 2)

    datas = []
    for data, label in zip(datas_word, labels):
        temp = []
        for word in data:
            if word in dictionary:
                temp.append(dictionary[word])
            else:
                temp.append(1)
        datas.append([temp, label])
    return datas


def load_test(path, dictionary):
    reader = open(path, 'r', encoding='utf-8')
    datas_word = []
    labels = []
    words = []

    # seg model
    # segmentor = Segmentor()
    # segmentor.load(CWS_MODEL_PATH)

    for line in reader.readlines():

        try:
            _, label, sentences = line.strip().split('\t')
            # sentences = " ".join(segmentor.segment(sentences))
        except:
            print("test line error", line)
            assert 1 == 2

        # if len(sentences) > 50:
        #     print(line)
        #     continue

        sentences = sentences.split()

        words.extend(sentences)
        # labels.append(label)

        if label in ['0', '4']:
            labels.append(1)
        elif label in ['1', '3']:
            labels.append(2)
        elif label in ['2', '5', '6', '7']:
            # elif label in ['-1']:
            labels.append(0)

        datas_word.append(sentences)

    datas = []
    for data, label in zip(datas_word, labels):
        temp = []
        for word in data:
            if word in dictionary:
                temp.append(dictionary[word])
            else:
                temp.append(1)
        datas.append([temp, label])
    return datas


def get_batch(datas, batch_size):
    """

    :param datas:
    :param batch_size:
    :return: <class 'tuple'>: ([[28, 402, 68, 51, 15, 708, 185, 860, 1610, 1, 11],...],
    array([1, 1, 1, 0, 1, 0, ...], dtype=int32))
    """
    random.shuffle(datas)
    x = []
    y = []
    for data in datas:
        sentence, label = data
        x.append(sentence)
        y.append(label)
        if len(x) >= batch_size:
            data_x = x
            data_y = y
            x = []
            y = []
            yield np.array(data_x), np.array(data_y, dtype=np.int32)
    if len(x) != 0:
        data_x = x
        data_y = y
        yield np.array(data_x), np.array(data_y, dtype=np.int32)


def pad(datas: object, seg_len: object = 100) -> object:
    """
    add zero
    :param datas:
    :param seg_len:
    :return:
    """
    result = []
    for data in datas:
        tmp = data[:seg_len] + [0] * max(seg_len - len(data), 0)
        result.append(tmp)
    return np.array(result, dtype=np.int32)


def get_test_data(datas):
    x = []
    y = []
    for data in datas:
        sentence, label = data
        x.append(sentence)
        y.append(label)
    return x, np.array(y, dtype=np.int32)


def load_dict(filename, dim):
    """
    load pretrain word embedding
    :param filename: the file that save word embedding
    :return: dictionary and word embedding. the i_th line of embedding is the i_th word in dictionary
    """
    dictionary = dict()
    dictionary_rev = dict()
    word_embeddings = []
    reader = open(filename, 'r', encoding='utf-8')

    random1 = np.random.randn(dim)
    word_embeddings.append([float(num) for num in list(random1)])
    random2 = np.random.randn(dim)
    word_embeddings.append([float(num) for num in list(random2)])

    for line in reader.readlines():
        line = line.strip().split(' ')
        dictionary[line[0]] = len(dictionary) + 2
        dictionary_rev[len(dictionary_rev) + 2] = line[0]
        # dictionary[line[0]] = len(dictionary)
        # dictionary_rev[len(dictionary_rev)] = line[0]
        embed = [float(num) for num in line[1:]]
        word_embeddings.append(embed)
    dictionary["PAD"] = 0
    dictionary["UNK"] = 1
    dictionary_rev[0] = "PAD"
    dictionary_rev[1] = "UNK"
    return dictionary, np.array(word_embeddings), dictionary_rev


sequence_length = 100
num_epochs = 5
keep_prob = 1.0
class_num = 3
batch_size = 64

cross_validate = False

embed_dim = 100
attention_size = 100
hidden_size = 100
layer_num = 3

# train_data_path = ['../data/train_rh_1215.csv']
train_data_path = ['../data/da12000train.seg',
                   '../data/da_highemotion_from_all_band.seg',
                   '../data/da_highemotion_from_datong1.seg',
                   '../data/da_highemotion_from_datong2.seg',
                   '../data/da_highemotion_from_datong3.seg'
                   # '../data/da_augmentation_1.seg'
                   # '../data/da_highemotion_from_datong4.seg',
                   # '../data/da_highemotion_from_datong5.seg'
                   ]
test_data_path = '../data/da5000test_from_datong.seg'
# test_data_path = '../data/valid_rh_1215.csv'

print("==============")
print("tittle:GRU+ATT,choice sentence with emotion")
print("sequence_length:", sequence_length)
print("keep_prob:", keep_prob)
print("batch_size:", batch_size)
print("attention_size:", attention_size)
print("hidden_size:", hidden_size)
print("layer_num:", layer_num)

print("==============")
print("cross_validate(8:2):", cross_validate, ".if True, ignore test data")
print("train_data_path:", train_data_path)
print("test_data_path:", test_data_path)

print("==============")
print("loading word2vec ...")
dictionary, word_embedding, dictionary_rev = load_dict('../data/program_vectors.txt', dim=embed_dim)

with open("../model/rnn/dictionary.pickle", "wb") as f1:
    pickle.dump(dictionary, f1)

with open("../model/rnn/word_embedding.pickle", "wb") as f2:
    pickle.dump(word_embedding, f2)

with open("../model/rnn/dictionary_rev.pickle", "wb") as f3:
    pickle.dump(dictionary_rev, f3)


print("loaded!")

if cross_validate:
    datas = load_train(train_data_path, dictionary)
    random.shuffle(datas)
    datas_size = len(datas)

    train_data = datas[:datas_size // 10 * 8]
    print("train data size:", len(train_data))

    test_data = datas[datas_size // 10 * 8:]
    print("test data size:", len(test_data))
else:
    train_data = load_train(train_data_path, dictionary)
    print("train data size:", len(train_data))

    test_data = load_test(test_data_path, dictionary)
    print("test data size:", len(test_data))

vocabulary_size = len(dictionary)


# Different placeholders
batch_ph = tf.placeholder(tf.int32, [None, sequence_length], name="batch_ph")
target_ph = tf.placeholder(tf.int64, [None], name="target_ph")
seq_len_ph = tf.placeholder(tf.int32, [None], name="seq_len_ph")
keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")

# Embedding layer
# embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, embed_dim], -1.0, 1.0), trainable=True, name="embeddings_var")
embeddings_var = tf.Variable(word_embedding, dtype=tf.float32, name="embeddings_var", trainable=False)

# saver = tf.train.Saver({"my_v2": embeddings_var})


batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph, name="batch_embedded")

# att embedding
att_embedding_out, _ = attention(batch_embedded, attention_size=30, name="att0", train_type=True)

# rnn dropout
fw = tf.contrib.rnn.DropoutWrapper(GRUCell(hidden_size), input_keep_prob=0.9)
bw = tf.contrib.rnn.DropoutWrapper(GRUCell(hidden_size), input_keep_prob=0.9)

# multi layers
multi_rnn_fw = rnn_cell.MultiRNNCell([fw] * layer_num)
multi_rnn_bw = rnn_cell.MultiRNNCell([bw] * layer_num)
# multi_rnn_fw = rnn_cell.MultiRNNCell([grid_rnn_cell.Grid1LSTMCell(num_units=hidden_size)] * layer_num)
# multi_rnn_bw = rnn_cell.MultiRNNCell([grid_rnn_cell.Grid1LSTMCell(num_units=hidden_size)] * layer_num)

# BiRNN layer
birnn_outputs, birnn_outputs_states = bidirectional_dynamic_rnn(
    multi_rnn_fw, multi_rnn_bw, inputs=batch_embedded,
    sequence_length=seq_len_ph, dtype=tf.float32)
# birnn_outputs, birnn_outputs_states = bidirectional_dynamic_rnn(rnn_cell.BasicLSTMCell(hidden_size), rnn_cell.BasicLSTMCell(hidden_size), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
# birnn_outputs, birnn_outputs_states = tf.nn.static_bidirectional_rnn(GRUCell(hidden_size), GRUCell(hidden_size), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# Attention layer
attention_output, WW = attention(birnn_outputs, attention_size, "att", train_type=True)

# merge att
output_result = tf.concat([att_embedding_out, attention_output], 1, name="output_result")

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph, name="drop")

# Fully connected layer
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, class_num], stddev=0.1), name="fully_w")
b = tf.Variable(tf.constant(0., shape=[class_num]), name="fully_b")
y_hat = tf.nn.tanh(tf.matmul(drop, W)+b, name="fully_y_hat")


# Cross-entropy loss and optimizer initialization
tv = tf.trainable_variables()
print("==============", tv)
regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=target_ph), name="loss")
loss = loss + regularization_cost

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, name="optimizer")

# Accuracy metric
y_predict = tf.argmax(y_hat, 1, name="y_predict")
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_predict, target_ph), tf.float32), name="accuracy")

init = tf.global_variables_initializer()

# Create a summary to monitor accuracy tensor


merged_summary_op = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(tf.global_variables())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)


def to_probability(x):  # str([ 0.95377445 -0.50079763 -0.98326391])
    while "[ " in x:
        x = x.replace("[ ", "[")
    x = eval(re.sub("[ ]+", ",", x))
    tmp = [math.exp(i) for i in x]
    return [i/sum(tmp) for i in tmp]


def test(epoch):
    data_x, data_y = get_test_data(test_data)
    data_x = pad(data_x)
    seq_len = np.array([len(x) for x in data_x])
    feed_dict = {batch_ph: data_x, target_ph: data_y, seq_len_ph: seq_len, keep_prob_ph: 1}
    y_hat_data, y_value, loss_value, accuracy_value = sess.run([y_hat, y_predict, loss, accuracy], feed_dict)

    from sklearn import metrics
    print(metrics.classification_report(data_y, y_value))
    print("accuracy: ", metrics.accuracy_score(data_y, y_value))
    print("confusion_matrix: \n", metrics.confusion_matrix(data_y, y_value))

    # save to file
    if not cross_validate:
        with open('../out/att_bilstm_result_%s.txt' % (str(epoch)), 'w', encoding='utf-8') as wo:
            with open(test_data_path, "r", encoding='utf-8') as ro:
                test_text = []
                sentence_id = []
                for line in ro:
                    line_spilt = line.split('\t')
                    # if len(line_spilt) < 3:
                    #     continue
                    sentence_id.append(line_spilt[0])
                    test_text.append(line_spilt[2])

                for id, y_true, y_pred, y_pro, text in zip(sentence_id, data_y, y_value, y_hat_data, test_text):
                    wo.write(str(id) + '\t' + str(y_true) + '\t' + str(y_pred) + '\t' + str(to_probability(str(y_pro))) + '\t' + str(text).strip() + '\n')

    # save to file when cross_validate
    # if cross_validate:
    #     with open('../out/att_bilstm_result_%s.txt' % (str(epoch)), 'w', encoding='utf-8') as wo:
    #         test_text = []
    #         for line in test_data:
    #
    #             test_text_sentence_code = line[0]
    #             text_tmp = [str(dictionary_rev[i]) for i in test_text_sentence_code]
    #             test_text.append(" ".join(text_tmp))
    #
    #         for y_true, y_pred, text in zip(data_y, y_value, test_text):
    #             wo.write(str(y_true) + '\t' + str(y_pred) + '\t' + str(text).strip() + '\n')


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        print(variable.name, "===", [dim.value for dim in shape])
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params
print("=========params num:", get_num_params())


total_step = len(train_data)/batch_size
step = 0
for epoch in range(num_epochs):

    batches = get_batch(train_data, batch_size)
    total_value = 0
    total_accuracy = 0
    for batch in batches:
        data_x, data_y = batch
        data_x = pad(data_x)
        seq_len = np.array([len(x) for x in data_x])
        # print(data_x, data_y)
        feed_dict = {batch_ph: data_x, target_ph: data_y, seq_len_ph: seq_len, keep_prob_ph: keep_prob}
        loss_value, accuracy_value, _ = sess.run([loss, accuracy, optimizer], feed_dict)
        # aaa = WW.eval(session=sess)
        # print(list(aaa))
        # save_path = saver.save(sess, "ttt.txt")

        total_value += loss_value
        total_accuracy += accuracy_value

    # Save the variables to disk.
    save_path = saver.save(sess, "../model/rnn/bilstm_att_%s" % (str(epoch)))
    print("Model saved in file: %s" % save_path)

    step += 1

    print('-----%d epoch Test-------' % (epoch))
    test(epoch)
