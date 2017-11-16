import sys, threading, time, os
import tensorflow as tf
from tensorflow.python.platform import googletest

import os
import tensorflow as tf
from tensorflow.python.framework import ops 
ops = tf.load_op_library('./libuser_ops.so') 

class CSVIterOpTest(tf.test.TestCase):
  def get_simple_format(self):
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = 100
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    return input_data_schema, feas, batch_size

  def testSimple(self):
    input_data_schema, feas, batch_size = self.get_simple_format()
    iter_op = ops.csv_iter('./sample_data.txt', input_data_schema, feas, batch_size=batch_size, label='label2')
    with self.test_session() as sess:
      label,sign = sess.run(iter_op)

      self.assertAllEqual(label.shape, [batch_size])
      self.assertAllEqual(sign.shape, [batch_size, len(feas)])
      self.assertAllEqual(sum(label), 2)
      self.assertAllEqual(sign[0,:], [7,0,4,1,1,1,5,9,8])

      label,sign = sess.run(iter_op)
      self.assertAllEqual(label.shape, [batch_size])
      self.assertAllEqual(sign.shape, [batch_size, len(feas)])
      self.assertAllEqual(sum(label), 1)
      self.assertAllEqual(sign[0,:], [9,9,3,1,1,1,5,4,8])

      for i in range(8):
        label,sign = sess.run(iter_op)
      label,sign = sess.run(iter_op)
      self.assertAllEqual(label.shape, [0])
      self.assertAllEqual(sign.shape, [0, len(feas)])

if __name__ == "__main__":
  googletest.main()
