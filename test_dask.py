import dask.dataframe as dd
import tensorflow as tf
import pandas as pd
import numpy as np
from pprint import pprint
import os
import tensorflow as tf
import time
import logging
ops = tf.load_op_library('./libuser_ops.so') 
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(funcName)s@%(filename)s#%(lineno)d %(levelname)s %(message)s')

def log_run_time(func):
    def do_func(*args, **kwargs):
        name = '%s(%s, %s)'%(func, ', '.join(map(str, args)), ', '.join([ '%s=%s'%(k, v) for k,v in kwargs.items() ]))
        logging.info('%s begins'%(name))
        t_begin = time.time()
        ret = func(*args, **kwargs)
        t_end = time.time()
        dt = t_end - t_begin
        logging.info('%s ends, time cost=%s seconds'%(name, dt))
        return ret 
    return do_func
def get_simple_format():
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = 10
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    label = 'label2'
    return input_data_schema, feas, batch_size, label





@log_run_time
def test_pandas(paths):
    schema, feas, batch_size, label = get_simple_format()
    buff = []
    for path in paths:
        data = pd.read_csv(path, names=schema, dtype={ fea:np.uint64 for fea in feas }, sep='\t')
        buff.append(data)
    data = pd.concat(buff).reset_index()
    return data
@log_run_time
def test_dask(paths):
    schema, feas, batch_size, label = get_simple_format()
    buff = []
    for path in paths:
        data = dd.read_csv(path, names=schema, dtype={ fea:np.uint64 for fea in feas }, sep='\t')
        buff.append(data)
    data = dd.concat(buff).reset_index()
    data = data.compute()
    return data

def main():
    data1 = test_pandas(['./sample_data.int48.txt' for i in range(2) ])
    data2 = test_dask(['./sample_data.int48.txt' for i in range(2) ])
    print (data1 - data2).sum()

    data1 = test_pandas(['./sample_data.int48.1m.txt' for i in range(2) ])
    data2 = test_dask(['./sample_data.int48.1m.txt' for i in range(2) ])
    print data1.shape, data2.shape
    print type(data1), type(data2)
    print (data1 - data2).sum()
    print data1.head()
    print data1.tail()
    print data2.head()
    print data2.tail()


if __name__ == '__main__':
    main()
