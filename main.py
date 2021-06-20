from configuration import *
import time
from subprocess import call

for run in range(start_run, start_run + total_runs):
    call('nohup sh SUPER{}_SETUP_WORLD.sh &'.format(super_tag), shell=True)
    print('loading run {}'.format(run))
    time.sleep(30)
    call('nohup sh SUPER{}_SETUP_PYTHON.sh {}'.format(super_tag,run), shell=True)
