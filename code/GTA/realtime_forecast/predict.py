#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
import numpy as np
from inference.config import *
from inference.utils import load_scaler, scaler_batch_data_predict, prepare_feed_data_predict
from online_utils import get_batch_data_real_time_predict
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def realtime_predict(flow_path,model_path):
    flow_scaler, date_scaler = load_scaler()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, model_path+"model.ckpt")
        graph = tf.get_default_graph()
        is_training = graph.get_operation_by_name('traffic_model/Placeholder').outputs[0]
        traffic_flow_input_data = graph.get_operation_by_name('traffic_model/Placeholder_1').outputs[0]
        targets = graph.get_operation_by_name('traffic_model/Placeholder_2').outputs[0]
        learning_rate = graph.get_operation_by_name('traffic_model/Placeholder_3').outputs[0]
        predict = graph.get_operation_by_name('traffic_model/Reshape_5').outputs[0]
        date, traffic_input = get_batch_data_real_time_predict(flow_path)
        test_batch_data = scaler_batch_data_predict(date, traffic_input, flow_scaler, date_scaler)
        test_date, test_traffic_input = prepare_feed_data_predict(test_batch_data)
        batch_test_date = np.tile(test_date,[BATCH_SIZE,1])
        batch_test_traffic_input = np.tile(test_traffic_input,[BATCH_SIZE,1,1])
        batch_predict = sess.run(predict,feed_dict={traffic_flow_input_data: batch_test_traffic_input})
        test_targets_scale = flow_scaler.inverse_transform(batch_predict)
        res_data = test_targets_scale[0]
        return_data = ','.join(str(i) for i in res_data)
        return return_data

