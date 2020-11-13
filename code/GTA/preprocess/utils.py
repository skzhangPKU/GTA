#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def load_file(file_name, conf):
    all_file_flow_list = []
    DATA_PATH = conf.get('config','flow_file_path')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line_strip = line.strip()
            with open(DATA_PATH+line_strip+".csv",'r') as file:
                item_list = file.readlines()
                all_item = []
                for item in item_list:
                    item_strip = item.strip().split(',')
                    all_item.append(item_strip)
                flow_list = np.array(all_item)[:,3]
                flow_float_list = [float(e) for e in flow_list]
                all_file_flow_list.append(flow_float_list)
    all_file_flow_arr = np.array(all_file_flow_list)
    return all_file_flow_arr

def write_test_to_tfrecords(ob_data_list, all_flow_label_list,n_shuffle_samples,conf):
    test_record_path = conf.get('config', 'test_record_path')
    test_writer = tf.python_io.TFRecordWriter(test_record_path)
    test_num = 0
    for k in range(n_shuffle_samples):
        ob_item = ob_data_list[k]
        ob_label = all_flow_label_list[k]
        ob_item_arr = np.array(ob_item).astype(np.float32)
        ob_label_arr = np.array(ob_label).astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ob_item_arr.tostring()])),
            'prediction_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ob_label_arr.tostring()]))
        }))
        serialized = example.SerializeToString()
        test_writer.write(serialized)
        test_num += 1
    test_writer.close()
    return test_num

def write_train_to_tfrecords(ob_data_list, all_flow_label_list,n_shuffle_samples,conf):
    train_record_path = conf.get('config', 'train_record_path')
    val_record_path = conf.get('config', 'val_record_path')
    train_writer = tf.python_io.TFRecordWriter(train_record_path)
    validation_writer = tf.python_io.TFRecordWriter(val_record_path)
    VALIDATION_PERCENTAGE = 20
    train_num, validation_num = 0,0
    for k in range(n_shuffle_samples):
        ob_item = ob_data_list[k]
        ob_label = all_flow_label_list[k]
        ob_item_arr = np.array(ob_item).astype(np.float32)
        ob_label_arr = np.array(ob_label).astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ob_item_arr.tostring()])),
            'prediction_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ob_label_arr.tostring()]))
        }))
        serialized = example.SerializeToString()
        chance = np.random.randint(100)
        if chance < VALIDATION_PERCENTAGE:
            validation_writer.write(serialized)
            validation_num += 1
        else:
            train_writer.write(serialized)
            train_num += 1
    validation_writer.close()
    train_writer.close()
    return train_num, validation_num