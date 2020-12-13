#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 12:31
@File:          dataset.py
'''

import pandas as pd
import numpy as np
import re
from jieba import posseg as pseg

def pre_text(text, chars=(',', '.', ';', '?', '!', '"', '\'', '(', ')', '{', '}', '[', ']', '+', '-', '*', '/',
                          '=', '&', '$', '#', '@', '%', '\\')):
    text = text.lower()

    text = text.replace('\'s', ' of ')

    for char in chars:
        text = text.replace(char, ' ')

    text = re.sub('\d+', 'num', text)

    text = re.sub(' +', ' ', text)

    return text

def remove_blank(term_list):
    for term in term_list:
        if term == ' ':
            term_list.remove(term)

def get_dataset(dataset_dir='D:/datasets/text_classification', stop_flag=('x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r')):
    train_dataset = pd.read_excel(f'{dataset_dir}/train.xlsx', usecols=['Class Index', 'Description'])
    val_dataset = pd.read_excel(f'{dataset_dir}/val.xlsx', usecols=['Class Index', 'Description'])

    X_train, Y_train, X_val, Y_val = [], [], [], []

    cnt = 0
    sample_len = len(train_dataset['Class Index'].values)
    for i in range(sample_len):
        text = train_dataset['Description'].values[i]
        cls_id = train_dataset['Class Index'].values[i]

        if text is None or not isinstance(text, str) or text == '':
            break

        if cls_id is None or not isinstance(cls_id, np.int64):
            break

        if cnt == 15000:
            break
        cnt += 1

        #text = pre_text(text)
        #term_list = jieba.lcut(text)
        #remove_blank(term_list)

        text_seged = pseg.cut(text.lower())

        term_list = []
        for term, flag in text_seged:
            if flag not in stop_flag:
                term_list.append(term) if not term.isdigit() else term_list.append('num')

        X_train.append(term_list)
        Y_train.append(cls_id)

    cnt = 0
    sample_len = len(val_dataset['Class Index'].values)
    for i in range(sample_len):
        text = val_dataset['Description'].values[i]
        cls_id = val_dataset['Class Index'].values[i]

        if text is None or not isinstance(text, str) or text == '':
            break

        if cls_id is None or not isinstance(cls_id, np.int64):
            break

        if cnt == 5000:
            break
        cnt += 1

        #text = pre_text(text)
        #term_list = jieba.lcut(text)
        #remove_blank(term_list)

        text_seged = pseg.cut(text.lower())

        term_list = []
        for term, flag in text_seged:
            if flag not in stop_flag:
                term_list.append(term) if not term.isdigit() else term_list.append('num')

        X_val.append(term_list)
        Y_val.append(cls_id)

    return (X_train, Y_train), (X_val, Y_val)