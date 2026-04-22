import os
import sys
import json
from collections import defaultdict

import pika
import pika.exceptions

from logging_utils import if_verbose

RMQ_DEFAULT_CONNECTION_URL = 'amqp://guest:guest@rabbitmq:5672/%2F'
RMQ_AUTOINCREMENT_REQUESTS_QUEUE_NAME = 'autoincrement_requests'
RMQ_MAGIC_REPLY_QUEUE_NAME = 'amq.rabbitmq.reply-to'

class Autoincrement:
    def __init__(self, rmq_connection_url, verbosity=0):
        connection_parameters = pika.URLParameters(rmq_connection_url)
        self.connection = pika.BlockingConnection(connection_parameters)
        self.channel = self.connection.channel()
        self.channel.basic_consume(
            queue=RMQ_MAGIC_REPLY_QUEUE_NAME, 
            on_message_callback=self.on_reply, 
            auto_ack=True)
        self.verbosity = verbosity
        self.result = None

    def __call__(self, request):
        self.result = None
        properties = pika.spec.BasicProperties(
            reply_to=RMQ_MAGIC_REPLY_QUEUE_NAME, 
            delivery_mode=pika.DeliveryMode.Persistent
        )
        self.channel.basic_publish(
            exchange='', 
            routing_key=RMQ_AUTOINCREMENT_REQUESTS_QUEUE_NAME, 
            body=json.dumps(request).encode(),
            properties=properties)
        self.channel.start_consuming()
        
        result = self.result
        self.result = None
        return result

    def on_reply(self, ch, method, properties, body):
        if_verbose(self.verbosity, 3, lambda: print(f'on_reply: {method=}, {properties=}, len(body)={len(body)}'))
        if_verbose(self.verbosity, 1, lambda: print(f'Response={body.decode()}'))
        self.result = body.decode()
        self.channel.close()

    @staticmethod
    def list(rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL, verbosity=0):
        client = Autoincrement(rmq_connection_url, verbosity=verbosity)
        request = dict(method='list')
        keys = client(request)
        return json.loads(keys)

    @staticmethod
    def get(key, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL, verbosity=0):
        client = Autoincrement(rmq_connection_url, verbosity=verbosity)
        request = dict(method='get', key=key)
        return client(request)

    @staticmethod
    def set(key, value, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL, verbosity=0):
        client = Autoincrement(rmq_connection_url, verbosity=verbosity)
        request = dict(method='set', key=key, value=value)
        return client(request)

class AutoincrementServer:
    def __init__(self, storage_fname, rmq_connection_url, verbosity=0):
        self.autoincs = defaultdict(int)
        self.storage_fname = storage_fname
        self.verbosity = verbosity
        
        self.connection_parameters = pika.URLParameters(rmq_connection_url)
        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()
        queue = self.channel.queue_declare(
            queue=RMQ_AUTOINCREMENT_REQUESTS_QUEUE_NAME,
            durable=True,
            arguments={'x-single-active-consumer': True},
        )

        if os.path.exists(storage_fname):
            if_verbose(self.verbosity, 3, lambda: print(f'Loading keys from "{storage_fname}"'))
            
            with open(storage_fname, 'r') as f:
                loaded = json.load(f)
                self.autoincs.update(loaded)

                if self.verbosity >= 3:
                    for k, v in self.autoincs.items():
                        print(f'Loaded {k}={v}')
                
                if_verbose(self.verbosity, 1, lambda: print(f'Loaded {len(self.autoincs)} keys'))

    def run(self):
        self.channel.basic_qos(prefetch_count=1) # max 1 unacked message, i.e. serial processing
        self.channel.basic_consume(
            queue=RMQ_AUTOINCREMENT_REQUESTS_QUEUE_NAME, 
            on_message_callback=self.on_request, 
            auto_ack=False)
        self.channel.start_consuming()

    def on_request(self, ch, method, properties, body):
        if_verbose(self.verbosity, 3, lambda: print(f'on_message: method={method}, properties={properties}, len(body)={len(body)}'))
        reply_to = properties.reply_to

        is_dirty = False
        
        try:
            request = json.loads(body.decode())
    
            match request['method']:
                case 'list':
                    if_verbose(self.verbosity, 2, lambda: print(f'Listing keys'))
                    result = json.dumps(list(self.autoincs.keys()))
                    if_verbose(self.verbosity, 2, lambda: print(f'Returning {len(self.autoincs)} keys'))
                case 'get':
                    key = request['key']
                    if_verbose(self.verbosity, 2, lambda: print(f'Generating autoincrement for {key=}'))
                    self.autoincs[key] += 1
                    result = str(self.autoincs[key])
                    if_verbose(self.verbosity, 2, lambda: print(f'Generated autoincrement for {key=} is {self.autoincs[key]}'))
                    is_dirty = True
                case 'set':
                    key = request['key']
                    value = int(request['value'])
                    if_verbose(self.verbosity, 2, lambda: print(f'Setting autoincrement for {key=} to {value}'))
                    self.autoincs[key] = value
                    result = str(self.autoincs[key])
                    if_verbose(self.verbosity, 2, lambda: print(f'Set autoincrement for {key=} to {self.autoincs[key]}'))
                    is_dirty = True
                case _:
                    assert False, f'Unsupported {request['method']=}'
        except Exception as ex: 
            result = 'error'
            if_verbose(self.verbosity, 0, lambda: print(f'Failed to handle request "{body.decode()[:1000]}": {str(ex)}'))
        
        if self.storage_fname and is_dirty:
            with open(self.storage_fname, 'w') as f:
                json.dump(self.autoincs, f)
                if_verbose(self.verbosity, 1, lambda: print(f'Saved {len(self.autoincs)} keys to "{self.storage_fname}"'))

        properties = pika.spec.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
        self.channel.basic_publish(
            exchange='', 
            routing_key=reply_to, 
            body=result.encode(),
            properties=properties)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_fname', type=str, default='')
    parser.add_argument('--rmq_connection_url', type=str, default='amqp://guest:guest@rabbitmq:5672/%2F')
    parser.add_argument('--verbosity', type=int, default=1)
    args = parser.parse_args()
    print(f'Running autoincrement server with args={args}')
    server = AutoincrementServer(args.storage_fname, args.rmq_connection_url, args.verbosity)
    server.run()