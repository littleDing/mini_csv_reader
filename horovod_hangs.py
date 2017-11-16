import horovod.tensorflow as hvd
import tensorflow as tf
import pandas as pd
import numpy as np
from pprint import pprint
import os
import tensorflow as tf
import time
import logging

def get_simple_format():
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = 10
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    label = 'label2'
    return input_data_schema, feas, batch_size, label

def iter_pandas_data(path, schema, feas, label, batch_size):
    batch_per_chunck = int(250*10000/batch_size)
    batch_per_chunck = max(1, batch_per_chunck)
    chunck_size = batch_per_chunck*batch_size
    diter = pd.read_table(path, names=schema,
                        engine='c', dtype = { fea:np.uint64 for fea in feas },
                        iterator=True, chunksize=chunck_size)
    for part in diter :
        label_buff = np.array(part[label])
        fea_buff = np.array([ np.array(part[col]%10) for col in feas ]).transpose()
        for i in range(batch_per_chunck):
              i_begin = i*batch_size
              i_end = (i+1)*batch_size
              if i_begin > part.shape[0]:
                  break
              label_batch = label_buff[i_begin:i_end]
              fea_batch = fea_buff[i_begin:i_end]
              yield label_batch, fea_batch

def get_logit():
    input_data_schema, feas, batch_size, label = get_simple_format()
    with tf.Graph().as_default():
        tf_label = tf.placeholder(tf.float32)
        tf_sign  = tf.placeholder(tf.int64)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
