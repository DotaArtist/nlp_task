#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""checkpoint 格式转换"""

__author__ = 'yp'


"""
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
"""