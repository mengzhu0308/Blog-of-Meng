#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 16:28
@File:          cnn.py
'''

from gensim import corpora
from keras.layers import Input
from keras import Model
from keras.optimizers import Adam
from keras import backend as K

from dataset import get_dataset
from generator import *
from utils import str2id, sequence_padding
from cnn_model import CNN_Model
from Loss import Loss
from Evaluate import Evaluate

class CrossEntropy(Loss):
    def compute_loss(self, inputs):
        y_true, y_pred = inputs
        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        return K.mean(loss)

if __name__ == '__main__':
    num_classes = 4
    vocab_size = 27795
    max_length = 40
    hidden_dim = 256
    train_batch_size = 64
    val_batch_size = 500

    (X_train, Y_train), (X_val, Y_val) = get_dataset()
    X = X_train + X_val
    dictionary = corpora.Dictionary(X)

    X_train = [str2id(x, dictionary.token2id) for x in X_train]
    X_val = [str2id(x, dictionary.token2id) for x in X_val]

    X_train = sequence_padding(X_train, max_length=max_length)
    Y_train = np.array(Y_train, dtype='int32') - 1
    X_val = sequence_padding(X_val, max_length=max_length)
    Y_val = np.array(Y_val, dtype='int32') - 1

    train_dataset = Dataset(X_train, Y_train)
    val_dataset = Dataset(X_val, Y_val)
    train_generator = generator(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_generator = generator(val_dataset, batch_size=val_batch_size, shuffle=False)

    text_input = Input(shape=(max_length, ), name='text_input', dtype='int32')
    y_true = Input(shape=(1, ), dtype='int32')
    out = CNN_Model(text_input, vocab_size=vocab_size, hidden_dim=hidden_dim, max_length=max_length,
                    num_classes=num_classes)
    out = CrossEntropy(-1)([y_true, out])
    model = Model([y_true, text_input], out)
    opt = Adam()
    model.compile(opt)

    num_train_batches = len(Y_train) // train_batch_size
    if len(Y_train) % train_batch_size != 0:
        num_train_batches += 1
    num_val_batches = len(Y_val) // val_batch_size
    if len(Y_val) % val_batch_size != 0:
        num_val_batches += 1

    evaluate = Evaluate(val_generator, num_val_batches, len(Y_val))

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_batches,
        epochs=5,
        callbacks=[evaluate],
        shuffle=False,
        initial_epoch=0
    )