#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from inference.utils import get_batch,prepare_feed_data
import tensorflow as tf
import datetime
from encoder.flags import FLAGS
from inference.config import *

class AutoEncoder(object):
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess):
    self.__shape = shape
    self.__num_hidden_layers = len(self.__shape) - 2
    self.__variables = {}
    self.__sess = sess
    self._setup_variables()

  @property
  def shape(self):
    return self.__shape

  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def session(self):
    return self.__sess

  def __getitem__(self, item):
    return self.__variables[item]

  def __setitem__(self, key, value):
    self.__variables[key] = value

  def _setup_variables(self):
    with tf.name_scope("autoencoder_variables"):
      for i in range(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,name=name_w,trainable=True,dtype=tf.float32)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b,dtype=tf.float32)
        if i < self.__num_hidden_layers:
          self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                name=name_w + "_fixed",
                                                trainable=False,dtype=tf.float32)
          self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                name=name_b + "_fixed",
                                                trainable=False,dtype=tf.float32)
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,trainable=True,name=name_b_out)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables_to_init(self, n):
    assert n > 0
    assert n <= self.__num_hidden_layers + 1
    vars_to_init = [self._w(n), self._b(n)]
    if n <= self.__num_hidden_layers:
      vars_to_init.append(self._b(n, "_out"))
    if 1 < n <= self.__num_hidden_layers:
      vars_to_init.append(self._w(n - 1, "_fixed"))
      vars_to_init.append(self._b(n - 1, "_fixed"))
    return vars_to_init

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def pretrain_net(self, input_pl, n, is_target=False):
    assert n > 0
    assert n <= self.__num_hidden_layers
    last_output = input_pl
    for i in range(n - 1):
      w = self._w(i + 1, "_fixed")
      b = self._b(i + 1, "_fixed")
      last_output = self._activate(last_output, w, b)
    if is_target:
      return last_output
    last_output = self._activate(last_output, self._w(n), self._b(n))
    out = self._activate(last_output, self._w(n), self._b(n, "_out"),transpose_w=True)
    out = tf.maximum(out, 1.e-9)
    out = tf.minimum(out, 1 - 1.e-9)
    return out

  def supervised_net(self, input_pl,regularizer):
    last_output = input_pl
    for i in range(self.__num_hidden_layers + 1):
      w = self._w(i + 1)
      regularizer(w)
      b = self._b(i + 1)
      last_output = self._activate(last_output, w, b)
    return last_output

def training(loss, learning_rate, loss_key=None):
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step

def loss_x_entropy(output, target):
  with tf.name_scope("xentropy_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                    target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
      return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),name='xentropy_mean')

def main_unsupervised(sess,data_utils,place_holder_set,input_data,input_layer_num):
    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in range(num_hidden)]
    ae_shape = [input_layer_num] + ae_hidden_shapes + [FLAGS.num_classes]
    ae = AutoEncoder(ae_shape, sess)
    num_train = TRAIN_SAMPLE_NUMS_FIFTEEN // BATCH_SIZE
    learning_rates = {j: getattr(FLAGS,
                                 "pre_layer{0}_learning_rate".format(j + 1))
                      for j in range(num_hidden)}
    noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
             for j in range(num_hidden)}
    for i in range(len(ae_shape) - 2):
      n = i + 1
      with tf.variable_scope("pretrain_{0}".format(n)):
        input_ = input_data
        target_ = input_data
        layer = ae.pretrain_net(input_, n)
        with tf.name_scope("target"):
          target_for_loss = ae.pretrain_net(target_, n, is_target=True)
        loss = tf.reduce_mean(tf.square(layer - target_for_loss))
        temp = set(tf.global_variables())
        train_op, global_step = training(loss, learning_rates[i], i)
        op_varible = set(tf.global_variables()) - temp
        vars_to_init = ae.get_variables_to_init(n)
        vars_to_init = vars_to_init + list(op_varible)
        sess.run(tf.variables_initializer(vars_to_init))
        for step in range(FLAGS.pretraining_epochs * num_train):
          batch_data= get_batch(sess,data_utils[0],data_utils[1],data_utils[2])
          date, traffic_input, targets = prepare_feed_data(batch_data)
          _,loss_value = sess.run([train_op, loss],feed_dict={place_holder_set[0]:traffic_input,place_holder_set[1]:targets,place_holder_set[2]:True})
    return ae

def main_supervised(ae,inputs,regularizer):
    logits = ae.supervised_net(inputs,regularizer)
    mat = tf.reshape(logits,[-1,INPUT_SIZE,TIME_SERIES_STEP])
    return mat

