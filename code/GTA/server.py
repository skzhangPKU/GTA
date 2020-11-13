#!/usr/bin/env python
# -*- coding: utf-8 -*-
from spyne import Application
from spyne import rpc
from spyne import ServiceBase
from spyne import Iterable, Unicode
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
import numpy as np
from realtime_forecast.predict import realtime_predict

# step1: Defining a Spyne Service
class HelloWorldService(ServiceBase):
    @rpc(Unicode, _returns=Iterable(Unicode))
    def say_hello(self, file_name):
        data = np.random.randint(0, 900, [1,4487])
        res = realtime_predict(data)
        yield res

# step2: Glue the service definition, input and output protocols
soap_app = Application([HelloWorldService], 'spyne.examples.hello.soap',
                       in_protocol=Soap11(validator='lxml'),
                       out_protocol=Soap11())

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return '<h1>Hello, web!</h1>'

# step3: Wrap the Spyne application with its wsgi wrapper
wsgi_app = WsgiApplication(soap_app)

if __name__ == '__main__':
    import logging
    from wsgiref.simple_server import make_server
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)
    logging.info("listening to http://127.0.0.1:6789")
    logging.info("wsdl is at: http://localhost:6789/?wsdl")
    server = make_server('10.5.14.165', 6789, application)
    server.serve_forever()
