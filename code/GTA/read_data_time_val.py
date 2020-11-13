#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from inference.datagenerator import BatchDataGenerator
from inference.config import *
import datetime
import os
from inference.utils import get_batch,load_scaler,prepare_feed_data
os.environ['CUDA_VISIBLE_DEVICES']='7'

if __name__=='__main__':
    flow_scaler, date_scaler = load_scaler()
    train_generator = BatchDataGenerator([TRAIN_RECORD_FILE], TRAIN_SAMPLE_NUMS_FIFTEEN, shuffle=True)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        starttime = datetime.datetime.now()
        batch_data = get_batch(sess,train_generator,flow_scaler,date_scaler)
        date, traffic_input, targets = prepare_feed_data(batch_data)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds," batch data featched finished")
        coord.request_stop()
        coord.join(threads)
