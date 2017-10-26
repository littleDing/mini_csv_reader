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
def testing_hand_write_op(path='./sample_data.txt', debug=False):
    '''
    calling hand write cpp op code
    '''
    input_data_schema, feas, batch_size, label = get_simple_format()
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            iter_op = ops.csv_iter(path, input_data_schema, feas, batch_size=batch_size, label=label) 
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
        n_data = 0
        with tf.Session() as sess:
            sess.run(init_op)
            n_data = 0
            while True :
                #print '>>>>>>>>>>>>>> before run batch', batch_idx
                label,sign = sess.run(iter_op)
                #print '>>>>>>>>>>>>>> after run batch', batch_idx
                if len(label) == 0 : 
                    break
                if debug and n_data <= batch_size*3:
                    print sign[0]
                n_data += len(label)
    logging.info('finally %d data loaded'%(n_data))

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

@log_run_time
def testing_pandas(path='./sample_data.txt', debug=False):
    '''
    itering data using pandas 
    '''
    input_data_schema, feas, batch_size, label = get_simple_format()
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            tf_label = tf.placeholder(tf.float32)
            tf_sign  = tf.placeholder(tf.int64)
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]

        n_data = 0
        with tf.Session() as sess:
            sess.run(init_op)
            n_data = 0
            for label,sign in iter_pandas_data(path, input_data_schema, feas, label, batch_size):
                #print '>>>>>>>>>>>>>> before run batch', batch_idx
                label,sign = sess.run([tf_label, tf_sign], feed_dict={ tf_label:label, tf_sign:sign })
                #print '>>>>>>>>>>>>>> after run batch', batch_idx
                if len(label) == 0 : 
                    break
                if debug and n_data <= batch_size*3:
                    print sign[0]
                n_data += len(label)
    logging.info('finally %d data loaded'%(n_data))

def read_my_file_format(filename_queue, schema, feas, label):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [ np.array([0], dtype=np.int64) if col in feas else [1.0] for col in schema ]
    row = tf.decode_csv(value, record_defaults=record_defaults, field_delim='\t')
    label_value = row[schema.index(label)]
    feas_value = [ row[schema.index(col)]%10 for col in feas  ]
    return label_value, feas_value

def input_pipeline(filenames, batch_size, schema, feas, label, num_epochs=None):
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
  label_value, feas_value = read_my_file_format(filename_queue, schema, feas, label)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  label_batch, fea_batch = tf.train.batch(
      [label_value, feas_value], batch_size=batch_size, capacity=capacity)
  return label_batch, fea_batch


@log_run_time
def testing_tf_csv(path='./sample_data.txt', debug=False):
    '''
    itering data using csv module in tf
    following https://www.tensorflow.org/programmers_guide/threading_and_queues
    '''
    input_data_schema, feas, batch_size, label = get_simple_format()
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            tf_label, tf_sign = input_pipeline([path], batch_size, input_data_schema, feas, label=label) 
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
        n_data = 0
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            n_data = 0
            try :
                while not coord.should_stop():
                    #print '>>>>>>>>>>>>>> before run batch', batch_idx
                    label,sign = sess.run([tf_label, tf_sign])
                    #print '>>>>>>>>>>>>>> after run batch', batch_idx
                    if len(label) == 0 : 
                        break
                    if debug and n_data <= batch_size*3:
                        print sign[0]
                    n_data += len(label)
            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()
            coord.join(threads)
    logging.info('finally %d data loaded'%(n_data))

def my_input_fn(file_path, schema, feas, batch_size, label):
    record_defaults = [ np.array([0], dtype=np.int64) if col in feas else [1.0] for col in schema ]
    label_index = schema.index(label)
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, record_defaults, field_delim='\t')
        labels = parsed_line[schema.index(label)]
        features = { fea:parsed_line[schema.index(fea)]%10   for fea in feas }
        return features,labels
    dataset = tf.contrib.data.TextLineDataset(file_path).map(decode_csv)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

@log_run_time
def testing_tf_data(path='./sample_data.txt', debug=False):
    '''
    itering data using the tf.data interface
    following
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    https://www.tensorflow.org/programmers_guide/datasets
    '''
    input_data_schema, feas, batch_size, label = get_simple_format()
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            tf_sign, tf_label = my_input_fn(path, input_data_schema, feas, batch_size, label)
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
        n_data = 0
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            n_data = 0
            try :
                while not coord.should_stop():
                    #print '>>>>>>>>>>>>>> before run batch', batch_idx
                    label,sign = sess.run([tf_label, tf_sign])
                    #print '>>>>>>>>>>>>>> after run batch', batch_idx
                    if len(label) == 0 : 
                        break
                    if debug and n_data <= batch_size*3:
                        print [ sign[fea][0] for fea in feas ]
                    n_data += len(label)
            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()
            coord.join(threads)
    logging.info('finally %d data loaded'%(n_data))




@log_run_time
def warm_up(path):
    for line in open(path):
        pass


def main(argv):
    ## do the tiny tests first, it should take ~0.05 seconds for every call
    ## and you should see 4 sample output for every call
    testing_hand_write_op('./sample_data.int48.txt', debug=True)
    testing_pandas('./sample_data.int48.txt', debug=True)
    testing_tf_csv('./sample_data.int48.txt', debug=True)
    testing_tf_data('./sample_data.int48.txt', debug=True)

    # warm up big files, let linux load it into cache first
    warm_up('./sample_data.int48.1m.txt')
    # ~25 seconds 
    testing_hand_write_op('./sample_data.int48.1m.txt')
    # ~23 seconds
    testing_pandas('./sample_data.int48.1m.txt')
    # ~192 seconds
    testing_tf_csv('./sample_data.int48.1m.txt')
    # ~164 seconds
    testing_tf_data('./sample_data.int48.1m.txt')


if __name__ == '__main__':
    tf.app.run()
