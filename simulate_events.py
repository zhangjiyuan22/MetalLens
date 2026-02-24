import sys
from functions_cpu import *
# from functions_cpu_parallel import *
from getconfig import *
import time


if __name__ == "__main__":
    time_begin = time.time()

    ########## change here #########
    ## eventname should be consistent with the name of .cfg file want to use
    eventname = 'mock1'
    ########## change here #########

    config = getEventInfo(eventname)

    SimulateEvents(config)

    time_end = time.time()
    print('\nTotal cost: %.2f min'%((time_end-time_begin)/60))
