#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from encoder.AutoEncoder import main_unsupervised
from encoder.AutoEncoder import main_supervised
from inference.config import *
from inference.utils import weight_variable, bias_variable, prepare_sae_inputs,get_current_input,tf_dot,masked_mae_tf
import datetime
flags = tf.app.flags
FLAGS = flags.FLAGS

def attention(cur_flow_input,adj_mx):
    sv_trans = tf.transpose(adj_mx)
    vec_concat = tf.concat((cur_flow_input,sv_trans),axis=1)
    vec_concat_dim = tf.expand_dims(vec_concat, 2)
    aw = tf.get_variable('attention_weight',[BATCH_SIZE,3,INPUT_SIZE*2])
    ab = tf.get_variable('attention_bias',[BATCH_SIZE,3])
    vec = tf.tanh(tf.squeeze(tf.matmul(aw,vec_concat_dim))+ab)
    vec_soft = tf.nn.softmax(vec)
    vec_soft_dim = tf.expand_dims(vec_soft,2)
    return vec_soft_dim

class LSTMModel(object):
    def __init__(self, inputs,training,regularizer):
        def lstm_cell():
            # return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
            return tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        attn_cell = lstm_cell
        if training is not None:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), input_keep_prob=KEEP_PROB, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(LAYERS_NUM)], state_is_tuple=True)
        self.initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(TIME_SERIES_STEP):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, :, time_step], state)
                outputs.append(cell_output)
        output = tf.reshape(outputs[-1], [-1, HIDDEN_SIZE])
        with tf.variable_scope("fullyConnect"):
            weight = tf.get_variable("weight", [HIDDEN_SIZE, INPUT_SIZE*INPUT_SIZE])
            bias = tf.get_variable("bias", [INPUT_SIZE*INPUT_SIZE])
        if regularizer != None: tf.add_to_collection('losses', regularizer(weight))
        logits = tf.matmul(output, weight) + bias
        self.mat = tf.reshape(logits, [-1, INPUT_SIZE, INPUT_SIZE])

class SAEModel(object):
    def __init__(self,sess,td_utils,ph_set,input_data,regularizer):
        input_dim = input_data.get_shape().as_list()[1]
        self.ae = main_unsupervised(sess, td_utils, ph_set, input_data, input_dim)
        self.mat = main_supervised(self.ae,input_data,regularizer)

class TRAFFICModel(object):
    def __init__(self,sess,td_utils,regularizer,initializer,embedded_mx,sensor_mx):
        self.is_training = tf.placeholder(tf.bool)
        self.traffic_flow_input_data_lstm = tf.placeholder(tf.float32, [None, INPUT_SIZE, TIME_SERIES_STEP])
        self.traffic_flow_input_data_sae = tf.placeholder(tf.float32, [None, TIME_REQUIRE, INPUT_SIZE*TIME_SERIES_STEP])#'traffic_model/Placeholder_2:0'
        self.targets = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        self.learning_rate =  tf.placeholder(tf.float32, [])
        ph_set = [self.traffic_flow_input_data_sae,self.targets,self.is_training]
        sdae_inputs = prepare_sae_inputs(self.traffic_flow_input_data_sae)
        with tf.variable_scope("traffic_model_month", reuse=None, initializer=initializer):
            model_month = SAEModel(sess,td_utils,ph_set,sdae_inputs[0],regularizer)
            lstm_month = LSTMModel(model_month.mat, self.is_training, regularizer)
        with tf.variable_scope("traffic_model_week", reuse=None, initializer=initializer):
            model_week = SAEModel(sess,td_utils,ph_set,sdae_inputs[1],regularizer)
            lstm_week = LSTMModel(model_week.mat, self.is_training, regularizer)
        with tf.variable_scope("traffic_model_cur", reuse=None, initializer=initializer):
            model_current = SAEModel(sess,td_utils,ph_set,sdae_inputs[2],regularizer)
            lstm_current = LSTMModel(model_current.mat, self.is_training, regularizer)
        bias = tf.get_variable("bias", [INPUT_SIZE, ])
        cur_flow_input = get_current_input(self.traffic_flow_input_data_sae)
        currentFlow = tf.reshape(cur_flow_input, [BATCH_SIZE, 1, INPUT_SIZE])
        sensor_w = tf.get_variable("sensor_w", [1], initializer=initializer)
        sensor_mx_conv = tf.reshape(sensor_mx, [1, INPUT_SIZE, INPUT_SIZE, 1])
        sensor_mx_conv = tf.layers.conv2d(sensor_mx_conv, 1, 1, strides=1, name='conv1')
        sensor_mx = tf.squeeze(sensor_mx_conv)
        sensor_mx = tf.multiply(sensor_w, sensor_mx)
        normal_w = attention(cur_flow_input, embedded_mx)
        tempMat = tf.add(tf.multiply(normal_w[:,0:1,:],lstm_month.mat), tf.multiply(normal_w[:,1:2,:],lstm_week.mat))
        affineMat = tf.add(tf.multiply(normal_w[:,2:3,:],lstm_current.mat), tempMat)
        affineMat = tf.multiply(sensor_mx, affineMat)
        predict = tf_dot(currentFlow, affineMat) + bias
        self.predict = tf.reshape(predict, [BATCH_SIZE, INPUT_SIZE])
        regularization_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(tf.square(self.predict - self.targets))+regularization_loss
        # self.loss = masked_mae_tf(self.predict,self.targets)+regularization_loss
        if self.is_training is None:
            return
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), 5)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
