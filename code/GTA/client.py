#!/usr/bin/env python
# -*- coding: utf-8 -*-

from suds.client import Client
h_clt = Client('http://127.0.0.1:6789/?wsdl')
h_clt.options.cache.clear()

file_name = 'file.csv'
result = h_clt.service.say_hello(file_name)
print(result)