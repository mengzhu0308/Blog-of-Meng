#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/11/7 19:18
@File:          generator.py
'''


import numpy as np

class Dataset:
    def __init__(self, texts, labels, text_transform=None):
        self.texts = texts
        self.labels = labels
        self.__text_transform = text_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        if self.__text_transform is not None:
            text = self.__text_transform(text)

        return text, label

def generator(dataset, batch_size=64, shuffle=True):
    nb_examples = len(dataset)
    rd_index = np.arange(nb_examples)
    flag_examples = nb_examples // batch_size * batch_size

    i = 0
    while True:
        if i != nb_examples and i == flag_examples:
            batch_size = nb_examples - flag_examples

        batch_texts = np.empty((batch_size, 40), dtype='int32')
        batch_labels = np.empty((batch_size, ), dtype='int32')

        for b in range(batch_size):
            if shuffle and i == 0:
                np.random.shuffle(rd_index)
                dataset.texts = dataset.texts[rd_index]
                dataset.labels = dataset.labels[rd_index]

            batch_texts[b], batch_labels[b] = dataset[i]

            i = (i + 1) % nb_examples

        yield [batch_labels, batch_texts], None