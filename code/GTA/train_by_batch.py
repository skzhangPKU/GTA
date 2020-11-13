#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from inference.datagenerator import BatchDataGenerator
import os
import numpy as np
from inference.config import *
from inference.utils import get_batch,load_scaler,prepare_feed_data
os.environ['CUDA_VISIBLE_DEVICES']='7'

def main():
    flow_scaler, date_scaler = load_scaler()
    test_generator = BatchDataGenerator(['input/test.tfrecords'], 128, shuffle=False)
    test_batch_num = TEST_SAMPLE_NUMS_FIFTEEN // 128
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+'/model.ckpt-37401.meta')
    with tf.Session() as sess:
        saver.restore(sess, "model/model.ckpt-37401")
        graph = tf.get_default_graph()
        is_training = graph.get_operation_by_name('traffic_model/Placeholder').outputs[0]
        traffic_flow_input_data = graph.get_operation_by_name('traffic_model/Placeholder_1').outputs[0]
        targets = graph.get_operation_by_name('traffic_model/Placeholder_2').outputs[0]
        learning_rate = graph.get_operation_by_name('traffic_model/Placeholder_3').outputs[0]
        predict  = graph.get_operation_by_name('traffic_model/Reshape_2').outputs[0]
        loss = graph.get_operation_by_name('traffic_model/Mean').outputs[0]
        train_op = graph.get_operation_by_name('traffic_model/Adam')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_all = []
        for index in range(100):
            test_batch_data = get_batch(sess, test_generator, flow_scaler, date_scaler)
            test_date, test_traffic_input, test_targets = prepare_feed_data(test_batch_data)
            test_loss, _ = sess.run([loss, train_op],
                                                feed_dict={traffic_flow_input_data: test_traffic_input,
                                                           targets: test_targets,learning_rate:0.00001,
                                                           is_training: True})
            print("test loss ",test_loss)
        for index in range(test_batch_num):
            test_batch_data = get_batch(sess, test_generator, flow_scaler, date_scaler)
            test_date, test_traffic_input, test_targets = prepare_feed_data(test_batch_data)
            test_loss, test_predicts = sess.run([loss, predict],
                                                feed_dict={traffic_flow_input_data: test_traffic_input,
                                                           targets: test_targets,
                                                           is_training: False})
            test_predicts_scale = flow_scaler.inverse_transform(test_predicts)
            test_targets_scale = flow_scaler.inverse_transform(test_targets)
            rmse_loss = np.sqrt(((test_predicts_scale - test_targets_scale) ** 2).mean())
            mae_loss = mean_absolute_error(test_predicts_scale, test_targets_scale)
            mape_loss = np.mean(np.abs((test_targets_scale - test_predicts_scale) / test_targets_scale))
            loss_all.append(mae_loss)
        model_test_loss = sum(loss_all) / len(loss_all)
        print('test loss is %.3f' % (model_test_loss))
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    main()