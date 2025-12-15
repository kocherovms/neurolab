import os
import time
import logging
import logging.handlers

class Logging(object):
    def __init__(self):
        self.logger = logging.getLogger('kmslog')
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.hasHandlers():
            syslogHandler = logging.handlers.SysLogHandler(address='/dev/log', facility=logging.handlers.SysLogHandler.LOG_LOCAL0)
            syslogHandler.ident = 'kmstag:'
            self.logger.addHandler(syslogHandler)

        self.app_name = 'MAIN'
        self.prefix_stanzas = dict()
        self.prefix_stanzas_order = []
        self.prefix = ''
        self.is_enabled = True
        self.last_log_time = time.time()
        
    def __call__(self, s, with_duration=True):
        t = time.time()
        
        if self.is_enabled:
            msg = ' ' # without this space following 'PID:'... will be considered as a part of syslogtag by rsyslog, so separate forcibly
            msg += f'PID:{os.getpid():<10} APP:{self.app_name:<15}'
            msg += ' ' + self.prefix
            msg += ' ' if self.prefix else ''
            
            if with_duration:
                duration = max(0, t - self.last_log_time)
                msg += f'{duration:>9.3f} >> '
                
            msg += s
            self.logger.debug(msg)

        self.last_log_time = t

    def push_prefix(self, stanza_name, stanza_value):
        if stanza_name in self.prefix_stanzas:
            self.prefix_stanzas[stanza_name] = stanza_value
        else:
            self.prefix_stanzas[stanza_name] = stanza_value
            self.prefix_stanzas_order.append(stanza_name)

        self.update_prefix()

    def pop_prefix(self, stanza_name):
        if stanza_name in self.prefix_stanzas:
            del self.prefix_stanzas[stanza_name]

        try:
            self.prefix_stanzas_order.remove(stanza_name)
        except ValueError:
            pass

        self.update_prefix()

    def update_prefix(self):
        self.prefix = ''

        if self.prefix_stanzas_order:
            self.prefix = '[' + ','.join(map(lambda s: f'{s}={self.prefix_stanzas[s]}', self.prefix_stanzas_order)) + ']'

    def auto_prefix(self, stanza_name, stanza_value, *args):
        d = {stanza_name: stanza_value}
        
        if args:
            assert len(args) % 2 == 0, f'args count is not even: {len(args)}'
            
            for ind in range(0, len(args), 2):
                d[args[ind]] = args[ind+1] 
            
        return AutoLoggingPrefix(self, d)

class AutoLoggingPrefix:
    def __init__(self, logging, d):
        self.logging = logging
        self.d = d

        for k, v in d.items():
            self.logging.push_prefix(k, v)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.d:
            self.logging.pop_prefix(k)
            
        return False
