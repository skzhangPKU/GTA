#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from wsgiref.simple_server import make_server
from realtime_forecast.predict import realtime_predict
from realtime_forecast.train_batch import realtime_batch_train
import configparser
import time
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

def get_config(training):
    conf = configparser.ConfigParser()
    conf.read('conf_dev.ini')
    if training:
        dir = conf.get('config', 'train_dir')
    else:
        dir = conf.get('config', 'predict_dir')
    model_path = conf.get('config', 'model_path')
    return dir,model_path

def predict(file_name):
    dir,model_path = get_config(False)
    flow_path = dir+file_name
    res = realtime_predict(flow_path,model_path)
    return res

def train(file_name):
    dir, model_path = get_config(True)
    flow_path = dir + file_name
    mae_loss = realtime_batch_train(flow_path, model_path)
    return mae_loss

def application(environ, start_response) :
    print(environ['PATH_INFO'])
    start_response('200 OK', [('Content-Type', 'text/html;charset=utf-8')])
    body = '<h1>Hello, %s!</h1>'%(environ['QUERY_STRING'] or 'web')
    queryString = environ['QUERY_STRING']
    if(queryString!=''):
        query_list = environ['QUERY_STRING'].split('=')
        type = query_list[0]
        filename = query_list[1]
        if type == 'predict':
            start = time.clock()
            rs = predict(filename)
            end = time.clock()
            print("run time: %f s" % (end - start))
        elif type == 'train':
            rs_np = train(filename)
            rs = str(rs_np)
        return [rs.encode('utf-8')]

if __name__ == '__main__':
    httpd = make_server('tensorflow-gpu-0.tensorflow-gpu.default.svc.cluster.local', 8002, application)
    print('Serving HTTP on port 8000...')
    httpd.serve_forever()
