#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from inference.config import *
import pickle

def masked_mae_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def get_current_input(traffic_flow_input_data):
    cur_traffic_input_data = traffic_flow_input_data[:, 2, :]
    cur_traffic_input_data_reshape = tf.reshape(cur_traffic_input_data, [-1, INPUT_SIZE, TIME_SERIES_STEP])
    cur_flow = cur_traffic_input_data_reshape[:, :, -1]
    return cur_flow

def transformData(flow, flow_scaler):
    flow_list = []
    for step in range(TIME_SERIES_STEP):
        tras_flow_data = flow_scaler.transform(flow[:, :, step]).astype(np.float32)
        expand_dim_data = np.expand_dims(tras_flow_data,axis=2)
        flow_list.append(expand_dim_data)
    flow_concat = np.concatenate(flow_list,axis=2)
    return flow_concat

def prepare_feed_data(batch_data):
    trans_date = batch_data[0]
    trans_month_data_ed = np.expand_dims(batch_data[1],axis=1)
    trans_week_data_ed = np.expand_dims(batch_data[2],axis=1)
    trans_cur_data_ed = np.expand_dims(batch_data[3],axis=1)
    traffic_flow_input = np.concatenate([trans_month_data_ed,trans_week_data_ed,trans_cur_data_ed],axis=1)
    return trans_date, traffic_flow_input, batch_data[4]

def prepare_feed_data_predict(batch_data):
    trans_date = batch_data[0]
    trans_month_data_ed = np.expand_dims(batch_data[1],axis=1)
    trans_week_data_ed = np.expand_dims(batch_data[2],axis=1)
    trans_cur_data_ed = np.expand_dims(batch_data[3],axis=1)
    traffic_flow_input = np.concatenate([trans_month_data_ed,trans_week_data_ed,trans_cur_data_ed],axis=1)
    return trans_date, traffic_flow_input

def scaler_batch_data_copy(date, inputs, predicts, flow_scaler,date_scaler):
    trans_month_flow_reshape = np.reshape(inputs[0], [-1, INPUT_SIZE*TIME_SERIES_STEP])
    trans_week_flow_reshape = np.reshape(inputs[1], [-1, INPUT_SIZE * TIME_SERIES_STEP])
    trans_cur_flow_reshape = np.reshape(inputs[2], [-1, INPUT_SIZE * TIME_SERIES_STEP])
    return date, trans_month_flow_reshape, trans_week_flow_reshape, trans_cur_flow_reshape, predicts

def scaler_batch_data(date, inputs, predicts, flow_scaler,date_scaler):
    trans_date = date_scaler.transform(date)
    trans_month_flow = transformData(inputs[0], flow_scaler)
    trans_week_flow = transformData(inputs[1], flow_scaler)
    trans_cur_flow = transformData(inputs[2], flow_scaler)
    trans_month_flow_reshape = np.reshape(trans_month_flow, [-1, INPUT_SIZE*TIME_SERIES_STEP])
    trans_week_flow_reshape = np.reshape(trans_week_flow, [-1, INPUT_SIZE * TIME_SERIES_STEP])
    trans_cur_flow_reshape = np.reshape(trans_cur_flow, [-1, INPUT_SIZE * TIME_SERIES_STEP])
    trans_predicts = flow_scaler.transform(predicts).astype(np.float32)
    return trans_date, trans_month_flow_reshape, trans_week_flow_reshape, trans_cur_flow_reshape, trans_predicts

def scaler_batch_data_predict(date, inputs, flow_scaler,date_scaler):
    trans_date = date_scaler.transform(date)
    trans_month_flow = transformData(inputs[0], flow_scaler)
    trans_week_flow = transformData(inputs[1], flow_scaler)
    trans_cur_flow = transformData(inputs[2], flow_scaler)
    trans_month_flow_reshape = np.reshape(trans_month_flow, [-1, INPUT_SIZE*TIME_SERIES_STEP])
    trans_week_flow_reshape = np.reshape(trans_week_flow, [-1, INPUT_SIZE * TIME_SERIES_STEP])
    trans_cur_flow_reshape = np.reshape(trans_cur_flow, [-1, INPUT_SIZE * TIME_SERIES_STEP])
    return trans_date, trans_month_flow_reshape, trans_week_flow_reshape, trans_cur_flow_reshape

def get_batch(sess,generator,flow_scaler,date_scaler):
    date, inputs, predicts = generator.next_batch(sess)
    return scaler_batch_data(date,inputs,predicts,flow_scaler,date_scaler)

def load_scaler():
    with open(FLOW_SCALER_PATH, 'rb') as flow_file:
        flow_scaler = pickle.load(flow_file)
    with open(DATE_SCALER_PATH, 'rb') as date_file:
        date_scaler = pickle.load(date_file)
    return flow_scaler,date_scaler

def prepare_sae_inputs(inputs):
    wea_data_month, wea_data_week, wea_data_cur = tf.split(value=inputs, num_or_size_splits=TIME_REQUIRE, axis=1)
    diamension = wea_data_month.shape[2].value
    wea_data_month_squ, wea_data_week_squ, wea_data_cur_squ = tf.reshape(wea_data_month, [-1, diamension]),tf.reshape(wea_data_week, [-1, diamension]),tf.reshape(wea_data_cur, [-1, diamension])
    return wea_data_month_squ, wea_data_week_squ, wea_data_cur_squ

def np_dot(matrixA,matrixB):
    mat = []
    for i in range(BATCH_SIZE):
        mat.append(np.dot(matrixA[i,:,:],matrixB[i,:,:]))
    res = np.concatenate(mat,axis=0)
    return res

def tf_dot(matrixA,matrixB):
    mat = []
    for i in range(BATCH_SIZE):
        mat.append(tf.matmul(matrixA[i,:],matrixB[i,:,:]))
    res = tf.concat(mat,axis=0)
    return res
