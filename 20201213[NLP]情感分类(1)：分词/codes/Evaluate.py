'''
@Author:        ZM
@Date and Time: 2020/7/30 9:24
@File:          Evaluate.py
'''

import numpy as np
from keras.callbacks import Callback

class Evaluate(Callback):
    def __init__(self, val_generator, num_val_baches, num_examples):
        super(Evaluate, self).__init__()
        self.val_generator = val_generator
        self.num_val_batches = num_val_baches
        self.num_examples = num_examples

    def on_epoch_end(self, epoch, logs=None):
        total_loss = 0.
        total_corrects = 0

        for _ in range(self.num_val_batches):
            batch_data, _ = next(self.val_generator)
            val_loss, predict = self.model.test_on_batch(batch_data, y=None), self.model.predict_on_batch(batch_data)

            total_loss += val_loss
            total_corrects += np.sum(batch_data[0] == np.argmax(predict, axis=-1))

        val_loss = total_loss / self.num_val_batches
        val_acc = (total_corrects / self.num_examples) * 100

        print(f'val_loss = {val_loss:.5f}, val_acc = { val_acc:.2f}')
