from collections import namedtuple

# namedtuple for data export by sequencing subsystem
SensorInstance = namedtuple('SensorInstance', 
                            ['Index', 
                             'x', 
                             'y', 
                             'normal', 
                             'normal_vec', 
                             'sweep', 
                             'x2', 
                             'y2'])
