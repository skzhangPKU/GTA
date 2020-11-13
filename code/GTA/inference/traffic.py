#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from inference.model import TRAFFICModel
from inference.datagenerator import BatchDataGenerator
from inference.utils import get_batch,load_scaler,prepare_feed_data
from sklearn.metrics import mean_absolute_error
from inference.config import *
import pickle
import time
from tensorflow.contrib.layers import xavier_initializer
import datetime
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='7'

def main(_):
    flow_scaler, date_scaler = load_scaler()
    train_generator = BatchDataGenerator([TRAIN_RECORD_FILE], BATCH_SIZE, shuffle=True)
    val_generator = BatchDataGenerator([VAL_RECORD_FILE], BATCH_SIZE, shuffle=False)
    test_generator = BatchDataGenerator([TEST_RECORD_FILE], BATCH_SIZE, shuffle=False)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    td_utils = [train_generator,flow_scaler,date_scaler]
    test_batch_num = TEST_SAMPLE_NUMS_FIFTEEN // BATCH_SIZE
    with open(EMBEDDED_MX_PATH,'rb') as file:
        embedded_mx = pickle.load(file)
    embedded_mx = tf.convert_to_tensor(embedded_mx, dtype=tf.float32)
    with open(SENSOR_MX_PATH, 'rb') as file2:
        sensor_mx = pickle.load(file2)
    sensor_mx = tf.convert_to_tensor(sensor_mx, dtype=tf.float32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Data preprocessing, it will take several minutes.')
        with tf.variable_scope("traffic_model", reuse=None, initializer=initializer):
            train_model = TRAFFICModel(sess,td_utils, regularizer=regularizer,initializer=initializer,embedded_mx=embedded_mx,sensor_mx=sensor_mx)
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        min_value = 10000
        rmse_min_value = 10000
        mape_min_value = 10000
        index = 100000
        start = datetime.datetime.now()
        print('Start training, time: ', start)
        for i in range(TRAINING_STEPS):
            batch_data = get_batch(sess,train_generator,flow_scaler,date_scaler)
            date, traffic_input, targets = prepare_feed_data(batch_data)
            loss,_ = sess.run([train_model.loss,train_model.train_op],feed_dict={train_model.traffic_flow_input_data_sae: traffic_input,
                                                                                 train_model.targets: targets,
                                                                                 train_model.learning_rate:LEARNING_RATE,
                                                                                 train_model.is_training:True})
            if i % 100 == 0:
                print("After %d steps,train loss is %.8f" % (i,loss))
            if i % 500 == 0:
                val_batch_data = get_batch(sess,val_generator,flow_scaler,date_scaler)
                val_date, val_traffic_input, val_targets = prepare_feed_data(val_batch_data)
                val_loss = sess.run(train_model.loss,feed_dict={train_model.traffic_flow_input_data_sae: val_traffic_input,
                                                                train_model.targets: val_targets,
                                                                train_model.is_training:False})

                print("After %d steps,Validation loss is %.8f" % (i,val_loss))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i+1)
                mae_loss_all = []
                rmse_loss_all = []
                mape_loss_all = []
                all_prediction_results = []
                for test_batch_index in range(test_batch_num):
                    test_batch_data = get_batch(sess, test_generator, flow_scaler, date_scaler)
                    test_date, test_traffic_input, test_targets = prepare_feed_data(test_batch_data)
                    test_loss, test_predicts = sess.run([train_model.loss, train_model.predict],
                                                        feed_dict={train_model.traffic_flow_input_data_sae: test_traffic_input,
                                                                   train_model.targets: test_targets,
                                                                   train_model.is_training: False})
                    test_predicts_scale = flow_scaler.inverse_transform(test_predicts)
                    test_targets_scale = flow_scaler.inverse_transform(test_targets)
                    for n in range(BATCH_SIZE):
                            all_prediction_results.append((test_date[n], test_predicts_scale[n], test_targets_scale[n]))
                    rmse_loss = np.sqrt(((test_predicts_scale - test_targets_scale) ** 2).mean())
                    mae_loss = mean_absolute_error(test_predicts_scale, test_targets_scale)
                    mape_loss = np.mean(np.abs((test_targets_scale - test_predicts_scale) / test_targets_scale)) * 100
                    mae_loss_all.append(mae_loss)
                    rmse_loss_all.append(rmse_loss)
                    mape_loss_all.append(mape_loss)
                mae_model_test_loss = sum(mae_loss_all) / len(mae_loss_all)
                rmse_model_test_loss = sum(rmse_loss_all) / len(rmse_loss_all)
                mape_model_test_loss = sum(mape_loss_all) / len(mape_loss_all)
                if mape_model_test_loss < mape_min_value:
                    index  = i
                    min_value = mae_model_test_loss
                    rmse_min_value = rmse_model_test_loss
                    mape_min_value = mape_model_test_loss
                    with open('../input/gta_prediction_15_'+time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())+'.pkl', 'wb') as f_pre:
                        pickle.dump(all_prediction_results, f_pre)
                else:
                    all_prediction_results = []
                print('test mae loss is %.8f ' % model_test_loss)
                print('test rmse loss is %.8f' % rmse_model_test_loss)
                print('test mape loss is %.8f' % mape_model_test_loss)
        print('best index is %d' % index)
        print('min_mae_value is %.8f' % min_value)
        print('min_rmse_value is %.8f' % rmse_min_value)
        print('min_mape_value is %.8f' % mape_min_value)
        end = datetime.datetime.now()
        print('Training end, time: ', end)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    tf.app.run()
