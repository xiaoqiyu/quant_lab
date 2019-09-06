# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : rpyxqi@gmail.com
# @file      : decorators.py

import time
from quant_models.utils.logger import Logger
from threading import Thread
from functools import wraps
import multiprocessing.pool as mpp
import multiprocessing.pool as mpp
import threading

from functools import wraps
from math import factorial

logger = Logger('log.txt', 'INFO', __name__).get_log()


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        # logger.info('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        print('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        return result

    return timed


def limit(number):
    ''' This decorator limits the number of simultaneous Threads
    '''
    sem = threading.Semaphore(number)

    def wrapper(func):
        @wraps(func)
        def wrapped(*args):
            with sem:
                return func(*args)

        return wrapped

    return wrapper


def async(f):
    ''' This decorator executes a function in a Thread'''

    @wraps(f)
    def wrapper(*args, **kwargs):
        thr = threading.Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


def parallel_pool(func):
    def async_run(*args, **kwargs):
        vals = args[0]
        ret = []
        with mpp.ThreadPool(5) as pool:
            results = [pool.apply_async(func, [x], kwds=kwargs) for x in vals]
            for item in results:
                try:
                    ret.append(item.get())
                except Exception as e:
                    logger.error("parallel run with error:{0}".format(e))
        return ret

    return async_run


def parallel_process(func):
    def async_run(*args, **kwargs):
        pass


@parallel_pool
def sample_parallel_threads(vals, *args, **kwargs):
    print(args, kwargs, vals)
    return vals[0]
    # return vals * 2


if __name__ == '__main__':
    from pprint import pprint

    # main(range(1000))
    # pprint(DIC)

    # ret = calcula_fatorial(range(100))

    # print(sample_parallel(list(range(50))))
    # print(run_parallel(list(range(50))))

    ret = sample_parallel_threads([[1, 2], [3, 4]], test_keys='testing')
    print(ret)
