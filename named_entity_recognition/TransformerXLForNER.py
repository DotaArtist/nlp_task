#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

# TODO

from transformers import AutoConfig
from transformers import AutoModelWithLMHead


config = AutoConfig.from_pretrained("D:/model_file/transfo-xl-wt103")
model = AutoModelWithLMHead.from_config(config)

