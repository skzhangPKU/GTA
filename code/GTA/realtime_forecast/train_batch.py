#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from sklearn.metrics import mean_absolute_error
from inference.config import *
from inference.utils import load_scaler,scaler_batch_data,prepare_feed_data
from online_utils import get_batch_data_realtime_train
os.environ['CUDA_VISIBLE_DEVICES']='7'

def realtime_batch_train(file_name,model_path):
    flow_scaler, date_scaler = load_scaler()
    EPOCH_NUM = 16
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
        predict  = graph.get_operation_by_name('traffic_model/Reshape_5').outputs[0]
        loss = graph.get_operation_by_name('traffic_model/Mean').outputs[0]
        train_op = graph.get_operation_by_name('traffic_model/Adam')
        date, traffic_input, predicts = get_batch_data_realtime_train(file_name)
        test_batch_data = scaler_batch_data(date, traffic_input, predicts, flow_scaler, date_scaler)
        test_date, test_traffic_input, test_targets = prepare_feed_data(test_batch_data)
        for i in range(EPOCH_NUM):
            test_loss, _ = sess.run([loss, train_op],
                                                feed_dict={traffic_flow_input_data: test_traffic_input,
                                                           targets: test_targets,learning_rate:0.001,
                                                           is_training: True})
            print("test loss ",test_loss)
        test_predicts = sess.run(predict,feed_dict={traffic_flow_input_data: test_traffic_input,is_training: False})
        test_predicts_scale = flow_scaler.inverse_transform(test_predicts)
        test_targets_scale = flow_scaler.inverse_transform(test_targets)
        mae_loss = mean_absolute_error(test_predicts_scale, test_targets_scale)
        saver.save(sess, os.path.join(model_path, 'model.ckpt'))
    return mae_loss


