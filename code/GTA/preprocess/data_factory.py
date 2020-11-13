#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import datetime
import tensorflow as tf
import configparser
from inference.datagenerator import BatchDataGenerator
from sklearn.preprocessing import MinMaxScaler
import pickle
from preprocess.utils import load_file,write_test_to_tfrecords,write_train_to_tfrecords
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
os.environ['CUDA_VISIBLE_DEVICES']='7'

def sample_list(flow_samples, sample_range, init_time,station_num,duration):
    fw_basic_time = int((7*4*24*3600)/duration)
    tw_basic_time = int((7*3*24*3600)/duration)
    ob_data_list = []
    all_flow_label_list = []
    for i in sample_range:
        row_data = []
        label_list = []
        time = init_time + datetime.timedelta(seconds=int((i + fw_basic_time + 6) * duration))  # 2688+6 当前时刻
        row_data.append(time.year)
        row_data.append(time.month)
        row_data.append(time.day)
        row_data.append(time.hour)
        row_data.append(time.minute)
        for num in range(station_num):
            traffic_flow = flow_samples[num]
            for j in range(6):
                row_data.append(traffic_flow[i+j])
            for j in range(6):
                row_data.append(traffic_flow[i+j+tw_basic_time])
            for j in range(6):
                row_data.append(traffic_flow[i +j + fw_basic_time])
            label_list.append(traffic_flow[i+6+fw_basic_time])
        ob_data_list.append(row_data)
        all_flow_label_list.append(label_list)
    return ob_data_list,all_flow_label_list

def tfrecords_from_data(flow_samples, init_time, conf, duration, flag):
    station_num = flow_samples.shape[0]
    n_samples = flow_samples.shape[1]
    n_shuffle_samples = n_samples - (int((24 * 3600 * 7 * 4)/duration) + 6 + 1)
    sample_range = np.array(range(n_shuffle_samples))
    ob_data_list, all_flow_label_list = sample_list(flow_samples, sample_range, init_time, station_num, duration)
    if flag:
        train_num, val_num = write_train_to_tfrecords(ob_data_list, all_flow_label_list, n_shuffle_samples, conf)
        conf.set('config', 'train_num', str(train_num))
        conf.set('config', 'val_num', str(val_num))
    else:
        test_num = write_test_to_tfrecords(ob_data_list, all_flow_label_list, n_shuffle_samples, conf)
        conf.set('config', 'test_num', str(test_num))
    return conf

def scalerMaxMin(train_num,conf):
    train_record_path = conf.get('config', 'train_record_path')
    flow_scaler_path = conf.get('config', 'flow_scaler_path')
    date_scaler_path = conf.get('config', 'date_scaler_path')
    train_gen = BatchDataGenerator([train_record_path], train_num, shuffle=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        date, month_flow, week_flow, cur_flow = train_gen.next_batch_fixed(sess)
        flow_data = month_flow[:, :, 0]
        flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_data)
        date_scaler = MinMaxScaler(feature_range=(0, 1)).fit(date)
        with open(flow_scaler_path, 'wb') as f1:
            pickle.dump(flow_scaler, f1)
        with open(date_scaler_path, 'wb') as f2:
            pickle.dump(date_scaler, f2)
        coord.request_stop()
        coord.join(threads)

def prepare_data(conf):
    file_name_path = conf.get('config','file_name_path')
    duration = conf.getint('config', 'duration')
    all_flow_samples = load_file(file_name_path,conf)
    all_samples = int(all_flow_samples.shape[1])
    conf.set('config', 'station_num', str(all_flow_samples.shape[0]))
    time_str = conf.get('config','init_time')
    time_list = list(map(int, time_str.split(',')))
    train_init_time = datetime.datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], 0)
    temp = int(((30+28)*24*60*60)/duration) + 5
    test_init_time = train_init_time+\
                     datetime.timedelta(seconds=(all_samples*duration))+\
                     datetime.timedelta(seconds=(-(temp*duration)))
    test_span = all_samples - temp + 1
    train_span = all_samples - int((30*24*60*60)/duration)
    train_samples = all_flow_samples[:,0:train_span]
    test_samples = all_flow_samples[:,test_span:]
    conf = tfrecords_from_data(train_samples,train_init_time,conf,duration,True)
    conf = tfrecords_from_data(test_samples,test_init_time,conf,duration,False)
    with open('conf_test.ini', 'w') as fw:
        conf.write(fw)
    train_num = conf.getint('config', 'train_num')
    scalerMaxMin(train_num,conf)

if __name__=='__main__':
    conf = configparser.ConfigParser()
    conf.read('conf_test.ini')
    prepare_data(conf)
