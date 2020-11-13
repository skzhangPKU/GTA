#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
from inference.config import *
from inference.utils import read_from_file, load_scaler, scaler_batch_data_predict, prepare_feed_data_predict
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def realtime_predict(file_name):
    flow_scaler, date_scaler = load_scaler()
    EPOCH_NUM = TRAINING_STEPS // (TRAIN_SAMPLE_NUMS_FIFTEEN // BATCH_SIZE)
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + '/model.ckpt-551.meta')
    with tf.Session() as sess:
        saver.restore(sess, "model/model.ckpt-551")
        graph = tf.get_default_graph()
        is_training = graph.get_operation_by_name('traffic_model/Placeholder').outputs[0]
        traffic_flow_input_data = graph.get_operation_by_name('traffic_model/Placeholder_1').outputs[0]
        targets = graph.get_operation_by_name('traffic_model/Placeholder_2').outputs[0]
        learning_rate = graph.get_operation_by_name('traffic_model/Placeholder_3').outputs[0]
        predict = graph.get_operation_by_name('traffic_model/predict').outputs[0]
        date, traffic_input, predicts = read_from_file(file_name)
        test_batch_data = scaler_batch_data_predict(date, traffic_input, flow_scaler, date_scaler)
        test_date, test_traffic_input = prepare_feed_data_predict(test_batch_data)
        batch_test_date = np.tile(test_date,[BATCH_SIZE,1])
        batch_test_traffic_input = np.tile(test_traffic_input,[BATCH_SIZE,1,1])
        batch_predict = sess.run(predict,feed_dict={traffic_flow_input_data: batch_test_traffic_input,
                                           is_training: True})
        res_data = batch_predict[0]
        return_data = ','.join(str(i) for i in res_data)
        return return_data

if __name__ == '__main__':
    realtime_predict('hdfs/file.csv')

