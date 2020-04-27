#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""transformer, bert, bilstm 实体抽取"""

__author__ = 'yp'

import re
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from functools import wraps
from sklearn.utils import shuffle
from bert_serving.client import BertClient
from datetime import timedelta, datetime
from sklearn.metrics import classification_report
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


BN_EPSILON = 0.001
BN_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'bn_update_ops'

ner_label_map = {
    'other': 0,
    'disease': 1, 'disease_i': 2,
    'symptom': 3, 'symptom_i': 4,
    'drug': 5, 'drug_i': 6,
    'diagnosis': 7, 'diagnosis_i': 8,
    'duration': 0, 'duration_i': 0,
    'start_time': 0, 'start_time_i': 0,
    'end_time': 0, 'end_time_i': 0
}


def count_macro_f1(a):
    p1 = a[0] / a[1]
    r1 = a[0] / a[2]
    p2 = a[3] / a[4]
    r2 = a[3] / a[5]
    p3 = a[6] / a[7]
    r3 = a[6] / a[8]
    p4 = a[9] / a[10]
    r4 = a[9] / a[11]
    p = (p1 + p2 + p3 + p4) / 4
    r = (r1 + r2 + r3 + r4) / 4
    return 2 * p * r / (p + r)


def count_weight_f1(a):
    p1 = a[0] / a[1]
    r1 = a[0] / a[2]
    p2 = a[3] / a[4]
    r2 = a[3] / a[5]
    p3 = a[6] / a[7]
    r3 = a[6] / a[8]
    p4 = a[9] / a[10]
    r4 = a[9] / a[11]
    p = (a[2] * p1 + a[5] * p2 + a[8] * p3 + a[11] * p4) / (a[2] + a[5] + a[8] + a[11])
    r = (a[2] * r1 + a[5] * r2 + a[8] * r3 + a[11] * r4) / (a[2] + a[5] + a[8] + a[11])
    return 2 * p * r / (p + r)


def get_disease_from_tag(sentence, tag, target=[1, 2]):
    sentence = list(sentence)
    out = []
    counter = 0
    for word, index in zip(sentence, tag):
        if counter == 1 and index not in target:
            counter = 0
            out.append(',')

        if index in target:
            counter = 1
            out.append(word)

    _ = ''.join(out)
    if len(_) > 1:
        _ = _[:-1]
        return _.split(',')
    else:
        return []


def get_disease_from_tag_new(sentence, tag, target=[1, 2]):
    """
    新增了位置信息的疾病抽取
    :param sentence: 输入句子
    :param tag: list, [start, middle, end]  [3,4,4] 数字根据标注约定的target，开始是3, 中间结束是3
    :param target: 如疾病的标注是[1，2]
    :return: [(_ner, _start_index, _end_index),(_ner, _start_index, _end_index),...] 同一个疾病会出现在多个位置，就有多个tuple
    """
    sentence = list(sentence)
    out = []
    counter = 0
    for word, index in zip(sentence, tag):
        if counter == 1 and index not in target:
            counter = 0
            out.append(',')

        if index in target:
            counter = 1
            out.append(word)

    _ = ''.join(out)
    if len(_) > 1:
        _ = _[:-1]  # 去除最后一次append的","
        ner_list = _.split(',')  # ner_list =['心脏病','糖尿病','心脏病',...] 根据上面sentence还有tag序列，把疾病集合抽取出来
    else:
        ner_list = []
    sentence = ''.join(sentence)

    def get_inner_position(_sentence, _tag, _target, _ner, _out_put_tuple=[], iter_num=0, iter_max=20):
        """
        寻找位置的函数，如一段话出现三个糖尿病，需要将三个糖尿病的位置都找出
        :param _sentence: 原句子
        :param _tag: [start, middle, end]  [3,4,4] 数字根据标注约定的target，开始是3, 中间结束是3
        :param _target: 如疾病的标注是[1，2]
        :param _ner: 糖尿病，具体的短语
        :param _out_put_tuple: [(_ner, _start_index, _end_index),(_ner, _start_index, _end_index),...] 同一个疾病会出现在多个位置，就有多个tuple
        :return:
        """
        try:
            _start_index = _sentence.index(_ner)  # 返回ner在sentence中出现的第一个位置
        except ValueError:
            return _out_put_tuple  # 找不到这个ner词的时候就返回，递归的中止条件
        if _start_index + len(_ner) < len(_sentence) - 1 and iter_num < iter_max:  # 如果找到的位置合理
            if tag[_start_index] == _target[0] \
                    and _tag[_start_index + len(_ner) + 1] != _target[1]:
                _end_index = _start_index + len(_ner)
            else:
                iter_num += 1
                return get_inner_position(_sentence=_sentence[_start_index + len(_ner):],
                                          _tag=_tag[_start_index + len(_ner):], _target=_target,
                                          _ner=_ner, _out_put_tuple=_out_put_tuple, iter_num=iter_num)
        else:
            _end_index = _start_index + len(_ner)

        # 强制否认
        try:
            if ('无' in _sentence[_start_index - 5: _start_index]) \
                    or ('否认' in _sentence[_start_index - 10: _start_index]):
                pass
            else:
                _out_put_tuple.append((_ner, _start_index, _end_index))
        except ValueError:
            _out_put_tuple.append((_ner, _start_index, _end_index))
            print("ner index error!!!")

        return get_inner_position(_sentence=_sentence[_start_index + len(_ner):],
                                  _tag=_tag[_start_index + len(_ner):], _target=_target,
                                  _ner=_ner, _out_put_tuple=_out_put_tuple, iter_num=iter_num)

    ner_list_with_position = []
    for ner in ner_list:
        if ner != "":
            ner_list_with_position.extend(get_inner_position(_sentence=sentence, _tag=tag, _target=target, _ner=ner))
    return list(set(ner_list_with_position))


def get_ner_label(sentence, target):
    if len(sentence) > MAX_LEN_SENTENCE:
        sentence = sentence[:MAX_LEN_SENTENCE]
    _label = [0 for _ in range(MAX_LEN_SENTENCE)]

    # target = target.split(':')[1]
    target = re.split(r'(ner_label):', target)[2]
    if target == 'null':
        return _label

    target_list = target.split('&&')
    for _target in target_list:
        _key, _value = _target.split('@@')

        try:
            ___ = re.finditer(str(_value), sentence)
        except:
            print(sentence)
        for m in re.finditer(_value, sentence):
            _label[m.start()] = ner_label_map[_key]

            if len(_value) > 1:
                _label[m.start() + 1: m.start() + len(_value)] = [ner_label_map[_key + '_i'] for _ in range(len(_value) - 1)]

    return _label


class DataProcess(object):
    def __init__(self, _show_token=False, feature_mode='remote'):
        self.bert_batch_size = 32
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.bert_model = BertPreTrain(mode=feature_mode)
        self.data_x = None
        self.data_y = None
        self.sentence_data = None

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            sent_list = []
            ner_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    try:
                        sent, ner = line.strip().strip("\n").split('\t')
                    except ValueError:
                        sent = line.strip().strip("\n").split('\t')
                        ner = 'NONE'

                    sent_list.append(sent)
                    ner_list.append(ner)

            data_tmp['sentence'] = pd.Series(sent_list)
            data_tmp['ner'] = pd.Series(ner_list)

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def get_feature(self):
        data_x = []
        data_y = []
        sentence_data = []

        _sentence_pair_list = []

        for index, row in tqdm(self.data.iterrows()):
            label = get_ner_label(row['sentence'], row['ner'])
            data_y.append(label)
            sentence_data.append(row['sentence'])

            _sentence_pair = row['sentence']
            _sentence_pair_list.append(_sentence_pair)

            if len(_sentence_pair_list) == 32:
                data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))
                _sentence_pair_list = []

        if len(_sentence_pair_list) > 0:
            data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        sentence_data = np.array(sentence_data)

        self.data_x = data_x
        self.data_y = data_y
        self.sentence_data = sentence_data

        print("data_x shape:", data_x.shape)
        print("data_y shape:", data_y.shape)
        print("sentence_data shape:", sentence_data.shape)

    def next_batch(self):
        counter = 0
        batch_x = []
        batch_y = []
        batch_sen = []

        for (_x, _y, _sen) in zip(self.data_x, self.data_y, self.sentence_data):
            if counter == 0:
                batch_x = []
                batch_y = []
                batch_sen = []

            batch_x.append(_x)
            batch_y.append(_y)
            batch_sen.append(_sen)

            counter += 1

            if counter == self.batch_size:
                counter = 0
                yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)
        yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)

    def get_one_sentence_feature(self, sentence):
        data_x = []
        data_y = []
        data_x.extend(list(self.bert_model.get_output([sentence], _show_tokens=False)))
        data_y.append([0 for _ in range(MAX_LEN_SENTENCE)])
        return np.array(data_x), np.array(data_y, dtype=np.int64)

    def get_sentence_list_feature(self, sentence_list):
        data_x = []
        data_y = []
        data_x.extend(list(self.bert_model.get_output(sentence_list, _show_tokens=False)))
        [data_y.append([0 for _ in range(MAX_LEN_SENTENCE)]) for i in sentence_list]
        return np.array(data_x), np.array(data_y, dtype=np.int64)


class Config(object):
    max_sentence_len = 100

    data_path = '../../data/medical_record/origin_data.tsv'
    output_data_path = '../../data/medical_record/final_extract_output.xlsx'
    train_data_path = '../../data/medical_record/train.txt'

    pattern_symptom = "./../data/medical_record/config/symptom_combine_all6.csv"
    pattern_drug = "./../data/medical_record/config/drug_combine.csv"
    pattern_check = "./../data/medical_record/config/check_combine.csv"

    pattern_disease = "./../data/medical_record/config/icd_10_disease.csv"
    pattern_operation = "./../data/medical_record/config/operation.csv"
    pattern_diagnosis = "./../data/medical_record/config/diagnosis.csv"
    pattern_composition = "./../data/medical_record/config/composition.csv"
    pattern_hospital = "./../data/medical_record/config/hospital_info.xlsx"

    pattern_hospital_cname = "./../data/medical_record/config/hospital_alias.xlsx"
    pattern_disease_cname = "./../data/medical_record/config/disease_cname.csv"

    pattern_hospital_add = "./../data/medical_record/config/hospital_add.xlsx"

    pre_train_embedding = "./../data/medical_record_character_embedding.txt"


MAX_LEN_SENTENCE = Config.max_sentence_len


class PreTrainProcess(object):
    def __init__(self, path=Config.pre_train_embedding, embedding_dim=256, sentence_len=100, pair_mode=False):
        embeddings = dict()

        self.embedding_path = path
        self.embedding_dim = embedding_dim
        self.sentence_len = sentence_len
        self.pair_mode = pair_mode

        with open(self.embedding_path, encoding='utf-8', mode='r') as f1:
            for line in f1.readlines():
                line = line.strip().split(' ')
                character = line[0]
                vector = [float(i) for i in line[1:]]

                if character not in embeddings.keys():
                    embeddings[character] = vector
        print('pre train feature loaded.')
        self.embedding_dict = embeddings

    def encode(self, sentence, **kwargs):
        if 'pair_mode' in kwargs.keys():
            if not isinstance(kwargs['pair_mode'], bool):
                raise TypeError("mode type must bool!")

        if 'pair_mode' in kwargs.keys() and kwargs['pair_mode']:
            try:
                assert isinstance(sentence, list)
            except AssertionError:
                print("sentence must be list!")
        else:
            try:
                assert isinstance(sentence, list)
                embedding_unk = [0.0 for _ in range(self.embedding_dim)]
                out_put = []

                for sentence_idx, _sentence in enumerate(sentence):
                    out_put_tmp = []

                    for char_idx, _char in enumerate(list(_sentence)):
                        if char_idx < self.sentence_len:
                            out_put_tmp.append(self.embedding_dict.get(_char, embedding_unk))

                    for i in range(self.sentence_len - len(out_put_tmp)):
                        out_put_tmp.append(embedding_unk)

                    out_put_tmp = np.stack(out_put_tmp, axis=0)
                    out_put.append(out_put_tmp)

                return np.stack(out_put, axis=0)
            except AssertionError:
                print("sentence must be list!")


class BertPreTrain(object):
    def __init__(self, mode='remote'):
        if mode == 'remote':
            self.model = BertClient(ip='192.168.236.14', port=5555, check_version=False)
        elif mode == 'pre_train':
            self.model = PreTrainProcess(path=Config.pre_train_embedding,
                                         embedding_dim=256, sentence_len=Config.max_sentence_len)
        else:
            self.model = BertClient(ip='127.0.0.1', port=5555, check_version=False)

    def get_output(self, sentence, _show_tokens=True):
        try:
            return self.model.encode(sentence, show_tokens=_show_tokens)
        except TypeError:
            print("sentence must be list!")


class LogParse(object):
    def __init__(self):
        self.delay = None
        self.logger = None
        self.handler = None
        self.filename = None
        self.path = None
        self.message = None

    def set_profile(self, path, filename, delay=0):
        self.path = path
        self.delay = delay

        today = datetime.today()
        mission_time = today + timedelta(days=-self.delay)
        mission_day = mission_time.strftime('%Y%m%d')

        self.filename = "{0}/{1}_{2}.logs".format(self.path, filename, mission_day)

        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(asctime)s:%(levelname)s:(info) %(message)s', level=logging.DEBUG)

        handler = logging.FileHandler(self.filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:(file) %(message)s'))

        logger.addHandler(handler)
        self.logger = logger
        self.handler = handler

    def exception(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # logs the exception
                err = "There was an exception in  "
                err += func.__name__
                self.logger.exception(error)

        return wrapper

    def info(self, message):
        self.message = message
        self.logger.info(self.message)


def count_macro_f1(a):
    p1 = a[0] / a[1]
    r1 = a[0] / a[2]
    p2 = a[3] / a[4]
    r2 = a[3] / a[5]
    p3 = a[6] / a[7]
    r3 = a[6] / a[8]
    p4 = a[9] / a[10]
    r4 = a[9] / a[11]
    p = (p1 + p2 + p3 + p4) / 4
    r = (r1 + r2 + r3 + r4) / 4
    return 2 * p * r / (p + r)


def count_weight_f1(a):
    p1 = a[0] / a[1]
    r1 = a[0] / a[2]
    p2 = a[3] / a[4]
    r2 = a[3] / a[5]
    p3 = a[6] / a[7]
    r3 = a[6] / a[8]
    p4 = a[9] / a[10]
    r4 = a[9] / a[11]
    p = (a[2] * p1 + a[5] * p2 + a[8] * p3 + a[11] * p4) / (a[2] + a[5] + a[8] + a[11])
    r = (a[2] * r1 + a[5] * r2 + a[8] * r3 + a[11] * r4) / (a[2] + a[5] + a[8] + a[11])
    return 2 * p * r / (p + r)


def get_disease_from_tag(sentence, tag, target=[1, 2]):
    sentence = list(sentence)
    out = []
    counter = 0
    for word, index in zip(sentence, tag):
        if counter == 1 and index not in target:
            counter = 0
            out.append(',')

        if index in target:
            counter = 1
            out.append(word)

    _ = ''.join(out)
    if len(_) > 1:
        _ = _[:-1]
        return _.split(',')
    else:
        return []


def get_disease_from_tag_new(sentence, tag, target=[1, 2]):
    """
    新增了位置信息的疾病抽取
    :param sentence: 输入句子
    :param tag: list, [start, middle, end]  [3,4,4] 数字根据标注约定的target，开始是3, 中间结束是3
    :param target: 如疾病的标注是[1，2]
    :return: [(_ner, _start_index, _end_index),(_ner, _start_index, _end_index),...] 同一个疾病会出现在多个位置，就有多个tuple
    """
    sentence = list(sentence)
    out = []
    counter = 0
    for word, index in zip(sentence, tag):
        if counter == 1 and index not in target:
            counter = 0
            out.append(',')

        if index in target:
            counter = 1
            out.append(word)

    _ = ''.join(out)
    if len(_) > 1:
        _ = _[:-1]  # 去除最后一次append的","
        ner_list = _.split(',')  # ner_list =['心脏病','糖尿病','心脏病',...] 根据上面sentence还有tag序列，把疾病集合抽取出来
    else:
        ner_list = []
    sentence = ''.join(sentence)

    def get_inner_position(_sentence, _tag, _target, _ner, _out_put_tuple=[], iter_num=0, iter_max=20):
        """
        寻找位置的函数，如一段话出现三个糖尿病，需要将三个糖尿病的位置都找出
        :param _sentence: 原句子
        :param _tag: [start, middle, end]  [3,4,4] 数字根据标注约定的target，开始是3, 中间结束是3
        :param _target: 如疾病的标注是[1，2]
        :param _ner: 糖尿病，具体的短语
        :param _out_put_tuple: [(_ner, _start_index, _end_index),(_ner, _start_index, _end_index),...] 同一个疾病会出现在多个位置，就有多个tuple
        :return:
        """
        try:
            _start_index = _sentence.index(_ner)  # 返回ner在sentence中出现的第一个位置
        except ValueError:
            return _out_put_tuple  # 找不到这个ner词的时候就返回，递归的中止条件
        if _start_index + len(_ner) < len(_sentence) - 1 and iter_num < iter_max:  # 如果找到的位置合理
            if tag[_start_index] == _target[0] \
                    and _tag[_start_index + len(_ner) + 1] != _target[1]:
                _end_index = _start_index + len(_ner)
            else:
                iter_num += 1
                return get_inner_position(_sentence=_sentence[_start_index + len(_ner):],
                                          _tag=_tag[_start_index + len(_ner):], _target=_target,
                                          _ner=_ner, _out_put_tuple=_out_put_tuple, iter_num=iter_num)
        else:
            _end_index = _start_index + len(_ner)

        # 强制否认
        try:
            if ('无' in _sentence[_start_index - 5: _start_index]) \
                    or ('否认' in _sentence[_start_index - 10: _start_index]):
                pass
            else:
                _out_put_tuple.append((_ner, _start_index, _end_index))
        except ValueError:
            _out_put_tuple.append((_ner, _start_index, _end_index))
            print("ner index error!!!")

        return get_inner_position(_sentence=_sentence[_start_index + len(_ner):],
                                  _tag=_tag[_start_index + len(_ner):], _target=_target,
                                  _ner=_ner, _out_put_tuple=_out_put_tuple, iter_num=iter_num)

    ner_list_with_position = []
    for ner in ner_list:
        if ner != "":
            ner_list_with_position.extend(get_inner_position(_sentence=sentence, _tag=tag, _target=target, _ner=ner))
    return list(set(ner_list_with_position))


def get_ner_label(sentence, target):
    if len(sentence) > MAX_LEN_SENTENCE:
        sentence = sentence[:MAX_LEN_SENTENCE]
    _label = [0 for _ in range(MAX_LEN_SENTENCE)]

    # target = target.split(':')[1]
    target = re.split(r'(ner_label):', target)[2]
    if target == 'null':
        return _label

    target_list = target.split('&&')
    for _target in target_list:
        _key, _value = _target.split('@@')

        try:
            ___ = re.finditer(str(_value), sentence)
        except:
            print(sentence)
        for m in re.finditer(_value, sentence):
            _label[m.start()] = ner_label_map[_key]

            if len(_value) > 1:
                _label[m.start() + 1: m.start() + len(_value)] = [ner_label_map[_key + '_i'] for _ in range(len(_value) - 1)]

    return _label


class DataProcess(object):
    def __init__(self, _show_token=False, feature_mode='remote'):
        self.bert_batch_size = 32
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.bert_model = BertPreTrain(mode=feature_mode)
        self.data_x = None
        self.data_y = None
        self.sentence_data = None

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            sent_list = []
            ner_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    try:
                        sent, ner = line.strip().strip("\n").split('\t')
                    except ValueError:
                        sent = line.strip().strip("\n").split('\t')
                        ner = 'NONE'

                    sent_list.append(sent)
                    ner_list.append(ner)

            data_tmp['sentence'] = pd.Series(sent_list)
            data_tmp['ner'] = pd.Series(ner_list)

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def get_feature(self):
        data_x = []
        data_y = []
        sentence_data = []

        _sentence_pair_list = []

        for index, row in tqdm(self.data.iterrows()):
            label = get_ner_label(row['sentence'], row['ner'])
            data_y.append(label)
            sentence_data.append(row['sentence'])

            _sentence_pair = row['sentence']
            _sentence_pair_list.append(_sentence_pair)

            if len(_sentence_pair_list) == 32:
                data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))
                _sentence_pair_list = []

        if len(_sentence_pair_list) > 0:
            data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        sentence_data = np.array(sentence_data)

        self.data_x = data_x
        self.data_y = data_y
        self.sentence_data = sentence_data

        print("data_x shape:", data_x.shape)
        print("data_y shape:", data_y.shape)
        print("sentence_data shape:", sentence_data.shape)

    def next_batch(self):
        counter = 0
        batch_x = []
        batch_y = []
        batch_sen = []

        for (_x, _y, _sen) in zip(self.data_x, self.data_y, self.sentence_data):
            if counter == 0:
                batch_x = []
                batch_y = []
                batch_sen = []

            batch_x.append(_x)
            batch_y.append(_y)
            batch_sen.append(_sen)

            counter += 1

            if counter == self.batch_size:
                counter = 0
                yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)
        yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)

    def get_one_sentence_feature(self, sentence):
        data_x = []
        data_y = []
        data_x.extend(list(self.bert_model.get_output([sentence], _show_tokens=False)))
        data_y.append([0 for _ in range(MAX_LEN_SENTENCE)])
        return np.array(data_x), np.array(data_y, dtype=np.int64)

    def get_sentence_list_feature(self, sentence_list):
        data_x = []
        data_y = []
        data_x.extend(list(self.bert_model.get_output(sentence_list, _show_tokens=False)))
        [data_y.append([0 for _ in range(MAX_LEN_SENTENCE)]) for i in sentence_list]
        return np.array(data_x), np.array(data_y, dtype=np.int64)


BN_EPSILON = 0.001
BN_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'bn_update_ops'


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)


def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=is_training)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=is_training)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def ln(inputs, epsilon=1e-8, scope="ln", *args):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)

    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs * masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = ln(outputs)

    return outputs


def feedforward(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


class TransformerCRF(object):
    def __init__(self, is_training=True, num_tags=9, learning_rate=0.0001,
                 embedding_size=256, sequence_length_val=100, keep_prob=0.9,
                 fc_hidden_num=200, bilstm_hidden_num=100, num_blocks=3,
                 num_headers=4, encoder_hidden_dim=256, feedfordward_hidden_dim=1024):
        self.sequence_length_val = sequence_length_val
        self.is_training = is_training
        self.num_tags = num_tags
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.bilstm_hidden_num = bilstm_hidden_num
        self.fc_hidden_num = fc_hidden_num

        self.num_blocks = num_blocks
        self.num_headers = num_headers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.feedfordward_hidden_dim = feedfordward_hidden_dim
        self.dropout_rate = 1 - keep_prob

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.sequence_length_val, self.embedding_size], name='input_x')
        self.input_y = tf.placeholder(tf.int64, shape=[None, None], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.transition_params = None

        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.decode_tags = self.predict_label()

    def inference(self):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = self.input_x
            enc *= self.encoder_hidden_dim ** 0.5  # scale

            enc += positional_encoding(enc, self.sequence_length_val)
            enc = tf.layers.dropout(enc, self.dropout_rate, training=self.is_training)

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_headers,
                                              dropout_rate=self.dropout_rate,
                                              training=self.is_training,
                                              causality=False)
                    enc = feedforward(enc, num_units=[self.feedfordward_hidden_dim, self.encoder_hidden_dim])  # bs * sl * ed
        memory = enc

        with tf.variable_scope('bilstm_layer', reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(self.bilstm_hidden_num)
            cell_bw = LSTMCell(self.bilstm_hidden_num)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=memory,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            bilstm_layer_output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            bilstm_layer_output = tf.nn.dropout(bilstm_layer_output, self.keep_prob)

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bilstm_hidden_num * 2, self.num_tags],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_tags],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            s = tf.shape(bilstm_layer_output)
            bilstm_layer_output = tf.reshape(bilstm_layer_output, [-1, self.bilstm_hidden_num * 2])

            fc_output = tf.nn.xw_plus_b(bilstm_layer_output, weights, biases)
            logits = tf.reshape(fc_output, [-1, s[1], self.num_tags])

        return logits

    def loss(self, ):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                    tag_indices=self.input_y,
                                                                    sequence_lengths=self.sequence_lengths)
        crf_loss = -tf.reduce_mean(log_likelihood)
        return crf_loss

    def predict_label(self):
        decode_tags, best_score = tf.contrib.crf.crf_decode(potentials=self.logits,
                                                            transition_params=self.transition_params,
                                                            sequence_length=self.sequence_lengths
                                                            )
        self.decode_tags = decode_tags
        return decode_tags

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op


a = LogParse()
a.set_profile(path="./", filename="exp")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'pre_train'
TRAIN_MODE = 'train'

model_dir = os.path.join('../', '_'.join([TransformerCRF.__name__, time.strftime("%Y%m%d%H%M%S")]))
a.info('model_dir:==={}'.format(model_dir))
print('model_dir:==={}'.format(model_dir))

if os.path.exists(model_dir):
    pass
else:
    os.mkdir(model_dir)

log_file = './log_file.txt'
_num = 0.00001

train_data_list = [
    '../data/normal_train/train_v3.txt',
]

test_data_list = ['../data/normal_train/test_v2.txt']

model = TransformerCRF(learning_rate=0.0001,
                       sequence_length_val=Config.max_sentence_len, num_tags=9)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if TRAIN_MODE == 'train':
    with open(log_file, mode='w', encoding='utf-8') as f1:
        with tf.Session(config=config) as sess:
            sess.run(init)

            train_data_process = DataProcess(feature_mode=FEATURE_MODE)
            train_data_process.load_data(file_list=train_data_list)
            train_data_process.get_feature()

            test_data_process = DataProcess(feature_mode=FEATURE_MODE)
            test_data_process.load_data(file_list=test_data_list, is_shuffle=False)
            test_data_process.get_feature()

            step = 0
            epoch = 40

            for i in range(epoch):

                for _, batch_x, batch_y in train_data_process.next_batch():
                    sum_counter = 0
                    right_counter = 0

                    model.is_training = True
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, _loss, _opt, transition_params = sess.run([model.logits,
                                                                        model.loss_val,
                                                                        model.train_op,
                                                                        model.transition_params
                                                                        ],
                                                                       feed_dict={model.input_x: batch_x,
                                                                                  model.input_y: batch_y,
                                                                                  model.sequence_lengths: _seq_len,
                                                                                  model.keep_prob: 0.8})

                    step += 1

                    for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        if viterbi_seq == list(_y_label):
                            right_counter += 1
                        sum_counter += 1

                    if step % 50 == 0:
                        a.info("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        print("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        f1.writelines("step:{} ===loss:{} === acc: {}\n".format(str(step),
                                                                                str(_loss),
                                                                                str(right_counter / sum_counter)))

                save_path = saver.save(sess, "%s/%s/model_epoch_%s" % (model_dir, str(i), str(i)))

                # test
                y_predict_list = []
                y_label_list = []

                sum_counter = 0
                right_counter = 0
                f1_statics = np.array([0 for i in range(12)])
                y_t = []
                y_p = []
                for batch_sentence, batch_x, batch_y in test_data_process.next_batch():
                    model.is_training = False
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, transition_params = sess.run([model.logits,
                                                           model.transition_params], feed_dict=
                                                          {model.input_x: batch_x,
                                                           model.input_y: batch_y,
                                                           model.sequence_lengths: _seq_len,
                                                           model.keep_prob: 1.0})
                    for _sentence_str, logit, seq_len, _y_label in zip(batch_sentence, _logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        y_p.extend(viterbi_seq)
                        y_t.extend(list(_y_label))

                        # predict
                        disease_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[1, 2])
                        symp_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[3, 4])
                        drug_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[5, 6])
                        diagnosis_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[7, 8])

                        # true
                        true_disease_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[1, 2])
                        true_symp_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[3, 4])
                        true_drug_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[5, 6])
                        true_diagnosis_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[7, 8])

                        # sum f1
                        f1_tmp = [len(set(disease_out).intersection(set(true_disease_out))),
                                  len(disease_out), len(true_disease_out)]
                        f1_tmp += [len(set(symp_out).intersection(set(true_symp_out))),
                                   len(symp_out), len(true_symp_out)]
                        f1_tmp += [len(set(drug_out).intersection(set(true_drug_out))),
                                   len(drug_out), len(true_drug_out)]
                        f1_tmp += [len(set(diagnosis_out).intersection(set(true_diagnosis_out))),
                                   len(diagnosis_out), len(true_diagnosis_out)]
                        f1_statics += np.array(f1_tmp)

                        if viterbi_seq == list(_y_label):
                            right_counter += 1
                        sum_counter += 1

                f1_statics = f1_statics.tolist()
                a.info("epoch: {}====f1_statics: {} \n".format(str(i), str(f1_statics)))
                f1.writelines("epoch: {}====f1_statics: {} \n".format(str(i), str(f1_statics)))

                precious = f1_statics[0] / f1_statics[1]
                recall = f1_statics[0] / f1_statics[2]
                print("epoch: {}====disease f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                a.info("epoch: {}====disease f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                f1.writelines(
                    "epoch: {}====disease f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))

                precious = f1_statics[3] / f1_statics[4]
                recall = f1_statics[3] / f1_statics[5]
                print("epoch: {}====symp f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                a.info("epoch: {}====symp f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                f1.writelines(
                    "epoch: {}====symp f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))

                precious = f1_statics[6] / f1_statics[7]
                recall = f1_statics[6] / f1_statics[8]
                print("epoch: {}====drug f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                a.info("epoch: {}====drug f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                f1.writelines(
                    "epoch: {}====drug f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))

                precious = f1_statics[9] / f1_statics[10]
                recall = f1_statics[9] / f1_statics[11]
                print("epoch: {}====diag f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                a.info("epoch: {}====diag f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))
                f1.writelines(
                    "epoch: {}====diag f1: {} \n".format(str(i), str(2 * precious * recall / (precious + recall + _num))))

                print("epoch: {}====: \n".format(str(i)), classification_report(y_pred=y_p, y_true=y_t))
                a.info("epoch: {}==={}=: \n".format(str(i), classification_report(y_pred=y_p, y_true=y_t)))
                f1.writelines("epoch: {}==={}=: \n".format(str(i), classification_report(y_pred=y_p, y_true=y_t)))

                a.info("epoch:{}==macro_f1:{}==weight_f1:{}".format(str(i),
                                                                    str(count_macro_f1(f1_statics)),
                                                                    str(count_weight_f1(f1_statics))))

                print("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))
                a.info("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))
                f1.writelines("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))

if TRAIN_MODE == 'predict':
    predict_data_list = ['../data/medical_record/train_3w.txt']

    predict_data_process = DataProcess(feature_mode=FEATURE_MODE)
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/19/model_epoch_19")

        y_predict_list = []
        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, transition_params = sess.run([model.logits,
                                                   model.transition_params],
                                                  feed_dict={model.input_x: batch_x,
                                                             model.sequence_lengths: _seq_len,
                                                             model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                y_predict_list.append(viterbi_seq)

        _out_file = predict_data_process.data
        _out_file['y_pred'] = pd.Series(y_predict_list)
        _out_file.to_csv('./final_predict.tsv', sep='\t')
