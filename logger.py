from time import time, sleep
from datetime import datetime
import sys

class Logger(object):
    def __init__(self, out_folder):
        timestamp = datetime.now()
        self.timestamp = '%d_%d_%d_%02.0d_%02.0d_%02.0d' % (
            timestamp.year, timestamp.month, timestamp.day, 
            timestamp.hour, timestamp.minute, timestamp.second
        )
        self.log_file = ('%s/log_%s.txt' % (out_folder, self.timestamp))
        
        with open(self.log_file, 'w') as f:
            print('Starting... \n', file=f)
        
    def log(self, *args, print_console=True, add_timestamp=True):
        timestamp = time()
        dt_obj = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ('%s:' % dt_obj, *args)
        
        successful = False
        max_attempts = 5
        attempt = 0
        while not successful and attempt < max_attempts:
            try: 
                with open(self.log_file, 'a+') as f:
                    print(*args, file=f)
                successful = True
            except IOError:
                print('%s: Failed to log: ' % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                attempt
        
        if print_console:
            print(*args)