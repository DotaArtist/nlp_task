#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""svm 文本分类，词袋模型"""

__author__ = 'yp'

import numpy as np
import pandas as pd
import dill as pickle
from sklearn import svm
from sklearn import metrics
from sklearn import calibration
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings('ignore')

label_map = {
    "0": 0,  # 其他
    "1": 1,  # 出院小结首页
    "2": 2,  # 出院小结其他页
    "3": 2,  # 入院记录首页
    "4": 2,  # 入院记录其他页
    "5": 3,  # 病案首页
    "6": 0,  # 专项检查（如心电图，脑电图、超声、CT、病理等）
}


def fun_map(x):
    return label_map[str(x)]


def fun_1(s):
    return str(s).split(" ")


def ngram_process(sentence):
    b = []
    gram_size = [1, 2, 3, 4]
    for i in range(len(sentence)):
        for j in gram_size:
            if i + j <= len(sentence):
                b.append(sentence[i:i + j])
    return " ".join(b)


def length_str(str_a):
    return len(str(str_a))


class TextClassifier(object):

    def __init__(self):
        self.train_data_path = None
        self.test_data_path = None

        self.model_path = None
        self.model_obj = None

        self.model_1 = None
        self.model_2 = None
        self.model_3 = None
        self.feature_names = None

        self.pre_str = None

    def train_main(self):
        data = pd.DataFrame()
        model_dict = dict()
        train_data_path = self.train_data_path

        for i in train_data_path:
            data_tmp = pd.read_excel(i, header=0)
            data_tmp.columns = ["pid", "label", "context"]

            data = pd.concat([data, data_tmp])

        data = shuffle(data)

        data["context_ngram"] = data[["context"]].applymap(ngram_process)
        context = data["context_ngram"].values

        label = data[["label"]].applymap(fun_map).values

        data_test = pd.read_excel(self.test_data_path, header=0)
        data_test.columns = ["pid", "label", "context"]

        data_test["context_ngram"] = data_test[["context"]].applymap(ngram_process)

        test_context = data_test["context_ngram"].values
        test_label = data_test[["label"]].applymap(fun_map).values

        # tf idf
        tf_idf = TfidfVectorizer(analyzer=fun_1, min_df=50)
        tf_idf.fit(context)

        model_dict["model_1"] = pickle.dumps(tf_idf)

        feature_names = tf_idf.get_feature_names()
        model_dict["feature_names"] = pickle.dumps(feature_names)
        print("feature num", len(feature_names))

        x_train = tf_idf.transform(context)
        x_test = tf_idf.transform(test_context)

        # chi
        model = SelectKBest(chi2, k="all")
        model.fit(x_train, label)

        model_dict["model_2"] = pickle.dumps(model)

        x_train = model.transform(x_train)
        x_test = model.transform(x_test)

        classify = svm.LinearSVC(C=0.9)

        # param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
        # grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
        # grid = xgb.XGBClassifier()
        # print(grid.best_params_)

        classify = calibration.CalibratedClassifierCV(classify, cv=10)

        classify.fit(x_train, label)
        y_predict = classify.predict(x_test)

        print(metrics.classification_report(test_label, y_predict))
        print("accuracy:", metrics.accuracy_score(test_label, y_predict))

        model_dict["model_3"] = pickle.dumps(classify)

        with open(self.model_path, mode='wb') as fm:
            joblib.dump(model_dict, fm)

    def load_model(self):
        with open(self.model_path, "rb") as fm:
            self.model_obj = pickle.load(fm)

            self.model_1 = pickle.loads(self.model_obj["model_1"])
            self.model_2 = pickle.loads(self.model_obj["model_2"])
            self.model_3 = pickle.loads(self.model_obj["model_3"])
            self.feature_names = pickle.loads(self.model_obj["feature_names"])

        result_temp = []
        for index in self.model_2.get_support(True):
            result_temp.append(self.feature_names[index])

    def predict_sentence(self, str_a):
        self.pre_str = str_a.replace("\t", " ")

        x_test = self.model_1.transform([ngram_process(self.pre_str)])
        x_test = self.model_2.transform(x_test)
        y_pro = self.model_3.predict_proba(x_test)
        return np.argmax(y_pro.reshape(-1))

    def predict_file(self, file_a, file_b):
        data_a = pd.read_excel(file_a, header=0)
        data_a.columns = ["pid", "label", "context"]
        data_a["map_label"] = data_a[["label"]].applymap(fun_map)
        data_a["predict"] = data_a[["context"]].applymap(self.predict_sentence)
        data_a.to_excel(file_b)


if __name__ == "__main__":
    md = TextClassifier()
    md.train_data_path = [
        './train.xlsx'
    ]
    md.test_data_path = './test.xlsx'
    md.model_path = './model.bin'

    print("training...")
    md.train_main()

    # print("loading model...")
    # md.load_model()

    # md.predict_file('./test.xlsx', './test_pred.xlsx')
