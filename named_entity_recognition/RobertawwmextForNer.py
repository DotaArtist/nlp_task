#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import torch
import json
import numpy as np
from transformers import XLNetTokenizer
from transformers import AutoConfig
from transformers import XLNetForTokenClassification

train_data = "D:/data_file/ccks2020_2_task1_train/task1_train.txt"
model_path = "D:/model_file/hfl_chinese-xlnet-base"
label_dict = {'疾病和诊断': 1, '影像检查': 3, '解剖部位': 5, '手术': 7, '药物': 9, '实验室检验': 11}


def transform_entities_to_label(text, entities, sep_sentence):
    char_label = np.array([0 for i in range(len(text))])
    out = np.array([0 for i in range(len(sep_sentence))])

    for i in entities:
        char_label[i["start_pos"]:i["end_pos"]] = label_dict[i["label_type"]]
        char_label[i["end_pos"]-1] = label_dict[i["label_type"]] + 1

    current_idx = 0
    for i, j in enumerate(sep_sentence[1:-2]):
        out[i+1] = max(char_label[current_idx:current_idx + len(j)])
        current_idx += len(j)

    return out.tolist()


config = AutoConfig.from_pretrained(model_path)
tokenizer = XLNetTokenizer.from_pretrained(model_path)
model = XLNetForTokenClassification.from_pretrained(model_path, num_labels=13)

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

with open(train_data, mode="r", encoding="utf-8") as f1:
    for line in f1.readlines():
        data = json.loads(line.strip())

        originalText = data["originalText"]
        entities = data["entities"]

        input_ids = torch.tensor(tokenizer.encode(originalText, add_special_tokens=True)).unsqueeze(0)
        sep_sentence = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
        seq_lab = transform_entities_to_label(originalText, entities, sep_sentence)
        labels = torch.tensor(seq_lab).unsqueeze(0)
        outputs = model(input_ids, labels=labels)
