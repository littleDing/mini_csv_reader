import tensorflow as tf
import pandas as pd
import numpy as np
from pprint import pprint
import os
import tensorflow as tf
import time
import logging
from threading import Thread, Lock, Semaphore
import Queue
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

G_BATCH_SIZE = 10
def get_simple_format():
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = G_BATCH_SIZE
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    label = 'label2'
    return input_data_schema, feas, batch_size, label

class SpeedTestCase(object):
    def run_single_thread(self, path, debug=False):
        input_data_schema, feas, batch_size, label = get_simple_format()
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                paths = [path] if type(path)==str else path
                self.prepare(paths, 1)
                init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
            with tf.Session() as sess:
                sess.run(init_op)
                self.set_session(sess)
                n_data = 0
                for label,sign in self.iter_one(sess, 0, 1):
                    if len(label) == 0 : 
                        break
                    if debug and n_data <= batch_size*3:
                        print sign[0]
                    n_data += len(label)
        logging.info('finally %d data loaded'%(n_data))
    def run_multi_thread(self, paths, n_thread, debug=False):
        session_config = tf.ConfigProto(
            intra_op_parallelism_threads = 0,
            inter_op_parallelism_threads = 0,
        )
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.prepare(paths, n_thread)
                init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]
            lock = Lock()
            n_data_holder = [0]
            def worker_thread(worker_idx):
                for label,sign in self.iter_one(sess, worker_idx, n_thread):
                    if len(label) == 0:
                        break
                    with lock :
                        n_data_holder[0] += len(label)
            with tf.Session(config=session_config) as sess:
                sess.run(init_op)
                self.set_session(sess)
                ts = []
                for wid in range(n_thread):
                    t = Thread(target=worker_thread, args=(wid,))
                    t.start()
                    ts.append(t)
                for t in ts:
                    t.join()
        logging.info('finally %d data loaded'%(n_data_holder[0]))
    def prepare(self, paths, n_thread):
        raise NotImplementedError()
    def set_session(self, sess):
        pass
    def iter_one(self, sess, worker_idx, total_worker):
        raise NotImplementedError()

class CaseHandwriteOp(SpeedTestCase):
    def prepare(self, paths, n_thread):
        input_data_schema, feas, batch_size, label = get_simple_format()
        self.iter_ops = [ ops.csv_iter(path, input_data_schema, feas, batch_size=batch_size, label=label)
                            for path in paths ] 
    def iter_one(self, sess, worker_idx, total_worker):
        for i in range(len(self.iter_ops)):
            if i % total_worker != worker_idx :
                continue
            iter_op = self.iter_ops[i]
            while True:
                label,sign = sess.run(iter_op)
                if len(label) == 0:
                    break
                yield label,sign
class CaseHandwriteOpWithTFQueue(SpeedTestCase):
    def __init__(self, n_thread=4):
        self.n_thread = n_thread
    def prepare(self, paths, n_thread):
        input_data_schema, feas, batch_size, label = get_simple_format()
        self.iter_ops = [ ops.csv_iter(path, input_data_schema, feas, batch_size=batch_size, label=label)
                            for path in paths ] 
        queue = tf.FIFOQueue(10, [tf.float32, tf.int64])
        self.tf_label = tf.placeholder(tf.float32)
        self.tf_sign  = tf.placeholder(tf.int64)
        self.enqueue_op = queue.enqueue([self.tf_label, self.tf_sign])
        self.dequeue_op = queue.dequeue()
        self.close_queue_op = queue.close()
        self.is_queue_close_op = queue.is_closed()
        self.queue_size_op = queue.size()
    def set_session(self, sess):
        sema = Semaphore(0)
        def producer(worker_idx, total_worker):
            for i in range(len(self.iter_ops)):
                if i % total_worker != worker_idx :
                    continue
                iter_op = self.iter_ops[i]
                while True:
                    label,sign = sess.run(iter_op)
                    if len(label) == 0:
                        break
                    feed_dict = {
                        self.tf_label : label,
                        self.tf_sign  : sign,
                    }
                    sess.run(self.enqueue_op, feed_dict=feed_dict)
            sema.release()
        for i in range(self.n_thread):
            t = Thread(target=producer, args=(i, self.n_thread))
            t.start()
        def queue_watchman():
            for i in range(self.n_thread):
                sema.acquire()
            while sess.run(self.queue_size_op) > 0 :
                time.sleep(1)
            sess.run(self.close_queue_op)
        Thread(target=queue_watchman).start()
    def iter_one(self, sess, worker_idx, total_worker):
        try :
            while not sess.run(self.is_queue_close_op):
                yield sess.run(self.dequeue_op)
        except tf.errors.OutOfRangeError:
            print 'iter data over'
class CaseHandwriteOpWithPYQueue(SpeedTestCase):
    def __init__(self, n_thread=4):
        self.n_thread = n_thread
    def prepare(self, paths, n_thread):
        input_data_schema, feas, batch_size, label = get_simple_format()
        self.iter_ops = [ ops.csv_iter(path, input_data_schema, feas, batch_size=batch_size, label=label)
                            for path in paths ] 
    def set_session(self, sess):
        self.queue = queue = Queue.Queue(self.n_thread)
        sema = Semaphore(0)
        def producer(worker_idx, total_worker):
            for i in range(len(self.iter_ops)):
                if i % total_worker != worker_idx :
                    continue
                iter_op = self.iter_ops[i]
                while True:
                    label,sign = sess.run(iter_op)
                    if len(label) == 0:
                        break
                    queue.put((label, sign))
            sema.release()
        for i in range(self.n_thread):
            t = Thread(target=producer, args=(i, self.n_thread))
            t.start()
        self.queue_closed = False
        def queue_watchman():
            for i in range(self.n_thread):
                sema.acquire()
            queue.join()
            self.queue_closed = True
        Thread(target=queue_watchman).start()
    def iter_one(self, sess, worker_idx, total_worker):
        try :
            while True:
                try :
                    label,sign = self.queue.get(timeout=0.1)
                    yield label,sign
                    self.queue.task_done()
                except Queue.Empty:
                    if self.queue_closed:
                        break
                    else :
                        print 'get data error'
        except tf.errors.OutOfRangeError:
            print 'iter data over'


@log_run_time
def testing_hand_write_op(path='./sample_data.txt', debug=False):
    '''
    calling hand write cpp op code
    '''
    CaseHandwriteOp().run_single_thread(path, debug) 


@log_run_time
def testing_hand_write_op_multithread(paths=['./sample_data.txt'], n_thread=1, debug=False):
    '''
    calling hand write cpp op code
    '''
    CaseHandwriteOp().run_multi_thread(paths, n_thread, debug) 
@log_run_time
def testing_hand_write_op_multithread_with_tfqueue(paths=['./sample_data.txt'], n_thread=1, debug=False):
    CaseHandwriteOpWithTFQueue(n_thread).run_single_thread(paths, debug) 
@log_run_time
def testing_hand_write_op_multithread_with_pyqueue(paths=['./sample_data.txt'], n_thread=1, debug=False):
    CaseHandwriteOpWithPYQueue(n_thread).run_single_thread(paths, debug) 

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
class CasePandas(SpeedTestCase):
    def prepare(self, paths, n_thread):
        self.tf_label = tf.placeholder(tf.float32)
        self.tf_sign  = tf.placeholder(tf.int64)
        self.paths = paths
    def iter_one(self, sess, worker_idx, total_worker):
        input_data_schema, feas, batch_size, label = get_simple_format()
        for i,path in enumerate(self.paths):
            if i % total_worker != worker_idx :
                continue
            for label,sign in iter_pandas_data(path, input_data_schema, feas, label, batch_size):
                label,sign = sess.run([self.tf_label, self.tf_sign], feed_dict={ 
                        self.tf_label:label, self.tf_sign:sign })
                yield label,sign

@log_run_time
def testing_pandas(path='./sample_data.txt', debug=False):
    '''
    itering data using pandas 
    '''
    CasePandas().run_single_thread(path, debug) 
@log_run_time
def testing_pandas_multithread(paths=['./sample_data.txt'], n_thread=1, debug=False):
    '''
    itering data using pandas 
    '''
    CasePandas().run_multi_thread(paths, n_thread, debug) 




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
    global G_BATCH_SIZE
    G_BATCH_SIZE = 10
    ## do the tiny tests first, it should take ~0.05 seconds for every call
    ## and you should see 3 sample output 
    testing_hand_write_op('./sample_data.int48.txt', debug=True)
    testing_pandas('./sample_data.int48.txt', debug=True)
    testing_tf_csv('./sample_data.int48.txt', debug=True)
    testing_tf_data('./sample_data.int48.txt', debug=True)
    testing_hand_write_op_multithread(['./sample_data.int48.txt']*4, n_thread=4)
    testing_pandas_multithread(['./sample_data.int48.txt']*4, n_thread=4)
    testing_hand_write_op_multithread_with_pyqueue(['./sample_data.int48.txt']*4, n_thread=4)
    testing_hand_write_op_multithread_with_tfqueue(['./sample_data.int48.txt']*4, n_thread=4)

    # warm up big files, let linux load it into cache first
    warm_up('./sample_data.int48.1m.txt')
    # ~25 seconds 
    #testing_hand_write_op('./sample_data.int48.1m.txt')
    # ~23 seconds
    #testing_pandas('./sample_data.int48.1m.txt')
    # ~192 seconds
    #testing_tf_csv('./sample_data.int48.1m.txt')
    # ~164 seconds
    #testing_tf_data('./sample_data.int48.1m.txt')

    # for 4-thread 
    # ~104 seconds
    #testing_hand_write_op_multithread(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~148 seconds 
    #testing_pandas_multithread(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~355 seconds with tf.queue
    #testing_hand_write_op_multithread_with_tfqueue(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~117 seconds with python queue
    #testing_hand_write_op_multithread_with_pyqueue(['./sample_data.int48.1m.txt']*4, n_thread=4)

    # for 8-thread 
    # ~214 seconds under default session config
    #testing_hand_write_op_multithread(['./sample_data.int48.1m.txt']*8, n_thread=8)
    # ~280 seconds under default session config
    #testing_pandas_multithread(['./sample_data.int48.1m.txt']*8, n_thread=8)

    G_BATCH_SIZE = 1000
    # ~2.82 for full-core-openmp, 2.3 seconds for 8core-openmp, 4s for 2core-openmp, 6.5 seconds for no openmp
    #testing_hand_write_op('./sample_data.int48.1m.txt')
    # ~3.07 seconds 
    #testing_pandas('./sample_data.int48.1m.txt')
    # ~2.3 seconds 
    #testing_hand_write_op('./sample_data.int48.1m.txt')
    # ~3.07 seconds 
    #testing_pandas('./sample_data.int48.1m.txt')

    # for 4-thread 
    # ~5.13 seconds for full-core-openmp
    # ~5.74 seconds for 2core
    # ~4.84 seconds for 4core
    # ~4.82 seconds for 8core
    # ~6.02 seconds for no openmp
    #testing_hand_write_op_multithread(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~4.56 seconds
    #testing_pandas_multithread(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~7.28 with python queue
    #testing_hand_write_op_multithread_with_pyqueue(['./sample_data.int48.1m.txt']*4, n_thread=4)
    # ~7.63 with tf queue
    #testing_hand_write_op_multithread_with_tfqueue(['./sample_data.int48.1m.txt']*4, n_thread=4)

    # for 8-thread 
    # ~6.2 seconds
    #testing_pandas_multithread(['./sample_data.int48.1m.txt']*8, n_thread=8)
    # ~6.9 seconds for no openmp
    #testing_hand_write_op_multithread(['./sample_data.int48.1m.txt']*8, n_thread=8)
    # ~7.89 seconds with python queue
    #testing_hand_write_op_multithread_with_pyqueue(['./sample_data.int48.1m.txt']*8, n_thread=8)
    # ~9.39 seconds with tf queue
    #testing_hand_write_op_multithread_with_tfqueue(['./sample_data.int48.1m.txt']*8, n_thread=8)

if __name__ == '__main__':
    tf.app.run()
