#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from inference.config import *
import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'

class BatchDataGenerator:
    def __init__(self, file_list, batch_size, shuffle):
        self.date,self.inputs,self.predicts = self.decode_from_tfrecords(file_name=file_list,batch_size=batch_size,shuffle=shuffle)

    def decode_from_tfrecords(self, file_name,shuffle, batch_size):
        filename_queue = tf.train.string_input_producer(file_name)
        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)
        example = tf.parse_single_example(serialized, features={
            'traffic_flow': tf.FixedLenFeature([], tf.string),
            'prediction_flow': tf.FixedLenFeature([], tf.string)
        })
        traffic_flow = tf.decode_raw(example['traffic_flow'],tf.float32)
        traffic_flow = tf.reshape(traffic_flow, [INPUT_SIZE*18+5])
        prediction_flow = tf.decode_raw(example['prediction_flow'],tf.float32)
        prediction_flow = tf.reshape(prediction_flow,[INPUT_SIZE])
        capacity = 50000
        min_after_dequeue = 10000
        if shuffle:
            inputs,predicts = tf.train.shuffle_batch([traffic_flow,prediction_flow], batch_size=batch_size,
                                                         num_threads=16, capacity=capacity, min_after_dequeue=min_after_dequeue) # 220 * 18(month 6, week 6, cur 6) + 5
        else:
            inputs, predicts = tf.train.batch([traffic_flow,prediction_flow], batch_size=batch_size, capacity=capacity)
        date = inputs[:,0:5]
        flow_station = inputs[:, 5:]
        fs_reshape = tf.reshape(flow_station, [-1, INPUT_SIZE, 18])
        traffic_input = fs_reshape[:, :,0:6], fs_reshape[:, :,6:12], fs_reshape[:, :, 12:]
        return date,traffic_input,predicts

    def next_batch(self, sess):
        return sess.run([self.date,self.inputs,self.predicts])

    def next_batch_fixed(self, sess):
        return sess.run([self.date,self.inputs[0],self.inputs[1],self.inputs[2]])
