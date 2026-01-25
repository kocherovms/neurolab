import os
import sys
import pickle
import io
import json
from functools import lru_cache

import pika
import pika.exceptions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg

import torch

RMQ_EVENTS_EXCHANGE_NAME = 'events'
RMQ_EVENTS_QUEUE_NAME = 'events'
RMQ_DEFAULT_CONNECTION_URL = 'amqp://guest:guest@rabbitmq:5672/%2F'

class RmqSummaryBase:
    def __init__(self, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL):
        self.connection_parameters = pika.URLParameters(rmq_connection_url)
        self.reconnect()
        exchange = self.channel.exchange_declare(
            exchange=RMQ_EVENTS_EXCHANGE_NAME, 
            exchange_type='topic',
            durable=True,
        )
        queue = self.channel.queue_declare(
            queue=RMQ_EVENTS_QUEUE_NAME,
            durable=True,
            arguments={'x-single-active-consumer': True},
        )
        self.channel.queue_bind(
            exchange=RMQ_EVENTS_EXCHANGE_NAME, 
            queue=RMQ_EVENTS_QUEUE_NAME,
        )

    def reconnect(self):
        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()
        
class RmqSummaryWriter(RmqSummaryBase):
    def __init__(self, log_dir, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL):
        super().__init__(rmq_connection_url)
        self.log_dir = log_dir

    def add_scalar(self, tag, scalar_value, global_step):
        properties = self._create_message_properties('add_scalar')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = str(global_step)
        
        with io.BytesIO() as b:
            if isinstance(scalar_value, torch.Tensor):
                scalar_value = scalar_value.item()
                
            pickle.dump(scalar_value, b)
            self._robust_publish(body=b.getvalue(), properties=properties)

    def add_text(self, tag, text_string, global_step):
        properties = self._create_message_properties('add_text')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = str(global_step)
        self._robust_publish(body=text_string.encode(), properties=properties)
        
    def add_figure(self, tag, figure, global_step, close):
        properties = self._create_message_properties('add_figure')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = str(global_step)

        with io.BytesIO() as b:
            image = self._figure_to_image(figure, close)
            pickle.dump(image, b)
            self._robust_publish(body=b.getvalue(), properties=properties)

    def add_hparams(self, hparam_dict, metric_dict, run_name):
        properties = self._create_message_properties('add_hparams')
        properties.headers['run_name'] = run_name

        message = dict(hparam_dict=hparam_dict, metric_dict=metric_dict)
        body = json.dumps(message)
        self._robust_publish(body=body.encode(), properties=properties)

    def flush(self):
        properties = self._create_message_properties('flush')
        self._robust_publish(body='', properties=properties)

    def _robust_publish(self, body, properties):
        for attempt_no in range(2):
            try:
                self.channel.basic_publish(
                    exchange=RMQ_EVENTS_EXCHANGE_NAME, 
                    routing_key=RMQ_EVENTS_QUEUE_NAME, 
                    body=body,
                    properties=properties)
                break
            except (pika.exceptions.StreamLostError, pika.exceptions.ConnectionClosedByBroker) as e:
                if attempt_no == 0:
                    self.reconnect()
                else:
                    raise

    def _figure_to_image(self, figure, close):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        
        if close:
            plt.close(figure)
        
        return image_chw

    def _create_message_properties(self, method):
        return pika.spec.BasicProperties(
            headers={
                'log_dir': self.log_dir,
                'method': method,
            },
            delivery_mode=pika.DeliveryMode.Persistent,
        )

class RmqSummaryCollector(RmqSummaryBase):
    def __init__(self, base_log_dir, rmq_connection_url):
        super().__init__(rmq_connection_url)
        self.base_log_dir = base_log_dir

    def run(self):
        self.channel.basic_qos(prefetch_count=1) # max 1 unacked message, i.e. serial processing
        self.channel.basic_consume(
            queue=RMQ_EVENTS_QUEUE_NAME, 
            on_message_callback=self.on_message, 
            auto_ack=False)
        self.channel.start_consuming()

    def on_message(self, ch, method, properties, body):
        # print(f'on_message: method={method}, properties={properties}, len(body)={len(body)}')
        logic_method = properties.headers['method']
        log_dir = properties.headers['log_dir']
        global_step = properties.headers.get('global_step', None)
        tag = properties.headers.get('tag', None)
        run_name = properties.headers.get('run_name', None)

        match logic_method:
            case 'add_scalar':
                with io.BytesIO(body) as b:
                    scalar_value = pickle.load(b)
                    self.get_summary_writer(log_dir).add_scalar(tag, scalar_value, global_step)
                    print(f'add_scalar, log_dir={log_dir}, tag={tag}, scalar_value={scalar_value}, global_step={global_step}')
            case 'add_text':
                text_string = body.decode()
                self.get_summary_writer(log_dir).add_text(tag, text_string, global_step)
                print(f'add_text, log_dir={log_dir}, tag={tag}, text_string="{text_string}", global_step={global_step}')
            case 'add_figure':
                with io.BytesIO(body) as b:
                    image_data = pickle.load(b)
                    self.get_summary_writer(log_dir).add_image(tag, image_data, global_step)
                    print(f'add_figure, log_dir={log_dir},  tag={tag}, shape(image_data)={image_data.shape}, global_step={global_step}')
            case 'add_hparams':
                with io.BytesIO(body) as b:
                    message = json.load(b)
                    hparam_dict = message['hparam_dict']
                    metric_dict = message['metric_dict']
                    print(f'add_hparams, log_dir={log_dir}, hparam_dict={hparam_dict}, metric_dict={metric_dict}, run_name={run_name}')
                    self.get_summary_writer(log_dir).add_hparams(hparam_dict, metric_dict, run_name=run_name)
            case 'flush':
                self.get_summary_writer(log_dir).flush()
                print('flush')
            case _:
                assert False, f'Unknown method="{logic_method}"'
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

    @lru_cache(maxsize=100)
    def get_summary_writer(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
        print(f'Creating SummaryWriter for log_dir={log_dir} (base_log_dir={self.base_log_dir})')
        return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_log_dir', type=str, default='/logdir')
    parser.add_argument('--rmq_connection_url', type=str, default='amqp://guest:guest@rabbitmq:5672/%2F')
    args = parser.parse_args()
    print(f'Running collector with args={args}')
    collector = RmqSummaryCollector(args.base_log_dir, args.rmq_connection_url)
    collector.run()
