# -*- coding: utf-8 -*-
# @time      : 2018/10/30 14:56
# @author    : rpyxqi@gmail.com
# @file      : f_buy_and_hold.py

from rqalpha.api import *


def init(context):
    context.s1 = "IF88"
    subscribe(context.s1)
    # logger.info("Interested in: " + str(context.s1))


def handle_bar(context, bar_dict):
    buy_open(context.s1, 1)


__config__ = {

    "base": {
        "start_date": "2019-01-09",
        "end_date": "2019-05-23",
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": None,
        "accounts": {
            "future": 1000000
        }
    },
    "extra": {
        "log_level": "error",
    },

    "mod": {
        "sys_progress": {
            "enabled": True,
            "show": True,
        },
    },

}
