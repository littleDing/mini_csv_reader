import tensorflow as tf
from pprint import pprint
import os
import tensorflow as tf
from tensorflow.python.framework import ops 
ops = tf.load_op_library('./libuser_ops.so') 

def get_simple_format():
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = 10
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    return input_data_schema, feas, batch_size

def testing_tf():
    path = './sample_data.txt'
    input_data_schema, feas, batch_size = get_simple_format()
    with tf.device('/cpu:0'):
        n_data_op = tf.placeholder(dtype=tf.float32)
        iter_op = ops.csv_iter(path, input_data_schema, feas, batch_size=batch_size, label='label2') 
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer() ]

    with tf.Session() as sess:
      sess.run(init_op)
      n_data = 0
      for batch_idx in range(3):
        print '>>>>>>>>>>>>>> before run batch', batch_idx
        ## it should be some debug printing here, but nothing come out when batch_idx>0
        label,sign = sess.run(iter_op)
        print '>>>>>>>>>>>>>> after run batch', batch_idx
        ## the content of sign remain the same every time
        print sign
        if len(label) == 0:
          break

def main(argv):
    testing_tf()















if __name__ == '__main__':
    tf.app.run()
