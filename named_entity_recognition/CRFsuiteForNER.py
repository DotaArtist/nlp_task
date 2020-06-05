#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'


from time import time
import pickle
import numpy as np

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn_crfsuite import metrics

from crfsuite import sent2features, sent2labels, sent2tokens
import scipy.stats


class DataPreprocess:
    def __init__(self, data_path, pre_tags, pre_tags_abbr):
        """
        data_path:str--->path of data
        pre_tags:list--->tag which should be made
        pre_tags_abbr:dict()--->abbreviation of pre_tags
        """
        self.data_path = data_path
        self.pre_tags = pre_tags
        self.pre_tags_abbr = pre_tags_abbr

    def process(self):
        start_time = time()
        data = []
        with open(self.data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                sentence = []
                sent_with_tag = line.split('\t')
                words = sent_with_tag[0].strip()
                tags = sent_with_tag[1].strip()
                tags_init = ['O' for _ in range(len(words))]
                word_with_tags = tags[tags.index(':') + 1:].strip()
                if word_with_tags == 'null':
                    pass
                else:
                    word_with_tag = word_with_tags.split('&&')
                    for w_t in word_with_tag:
                        try:
                            tag = w_t.split('@@')[0]
                            word = w_t.split('@@')[1]
                        except Exception:
                            #                             print('ner tag ERROR：{}'.format(line))
                            break
                        if tag in self.pre_tags:
                            tag_abbr = self.pre_tags_abbr[tag]
                        else:
                            tag_abbr = 'O'
                        cnt_word = words.count(word)
                        pos = 0
                        for i in range(cnt_word):
                            pos = words.index(word, pos)
                            for j in range(pos, pos + len(word)):
                                if tag_abbr == 'O':
                                    tags_init[j] = tag_abbr
                                else:
                                    if j == pos:
                                        tags_init[j] = 'B-' + tag_abbr
                                    else:
                                        tags_init[j] = 'I-' + tag_abbr
                for idx, char in enumerate(words):
                    sentence.append((char, tags_init[idx]))
                if len(sentence) == 0:
                    print(line)
                data.append(sentence)
            end_time = time()
            print('Data processing is over! It takes {} s.'.format(
                '%.2f' % (end_time - start_time)))
            return data


# 调参
def tune_parameters():
    labels = ['B-DIS', 'I-DIS', 'B-SYM', 'I-SYM', 'B-DIA', 'I-DIA', 'B-DRU', 'I-DRU', 'O']
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               max_iterations=100,
                               all_possible_transitions=True)
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        'all_possible_states': [True, False]
    }
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted',
                            labels=labels)
    rs = RandomizedSearchCV(crf,
                            params_space,
                            cv=10,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=100,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)

    # best params:{'all_possible_states': True, 'c1': 0.1697585004992326, 'c2': 0.01998358278071205}
    # best CV score:0.929676333861697
    # model size: 1.34M


# 训练、保存模型
def train_and_save():
    print('Training...')
    start_time = time()
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               c1=0.1698,
                               c2=0.0200,
                               max_iterations=100,
                               all_possible_states=True,
                               all_possible_transitions=True)
    crf.fit(X_train, y_train)
    end_time = time()
    print('Train is over! It takes {} s.'.format('%.2f' % (end_time - start_time)))

    # 保存模型:
    with open('mycrf.pickle', 'wb') as f:
        pickle.dump(crf, f)
    model_path = 'mycrf.pickle'
    return model_path


# 评估
def evaluate():
    from crfsuite import tag_to_entity_all
    with open('mycrf.pickle', 'rb') as f:
        crf = pickle.load(f)
    y_pred = crf.predict(X_test)
    word_f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=crf.classes_)
    cnt = 0
    y_pred_entity = tag_to_entity_all(y_pred)
    y_test_entity = tag_to_entity_all(y_test)
    for i in range(len(y_pred_entity)):
        if y_pred_entity[i] == y_test_entity[i]:
            cnt += 1
    sentence_accuracy = cnt / len(y_pred_entity)
    print('sentence_accuracy: {}, word_f1_score: {}'.format(sentence_accuracy, word_f1_score))
    # 0.6061830672863204 0.9325099689109188


if __name__ == '__main__':
    pre_tags = ['disease', 'symptom', 'diagnosis', 'drug']
    pre_tags_abbr = {
        'disease': 'DIS',
        'symptom': 'SYM',
        'diagnosis': 'DIA',
        'drug': 'DRU',
        'duration': 'DUR',
        'start_time': 'STA',
        'end_time': 'END'
    }
    dataprocess = DataPreprocess('data/ner_all_3w.txt', pre_tags, pre_tags_abbr)
    train_data = dataprocess.process()

    X = [sent2features(s) for s in train_data]
    y = [sent2labels(s) for s in train_data]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=0)

    # evaluate()
