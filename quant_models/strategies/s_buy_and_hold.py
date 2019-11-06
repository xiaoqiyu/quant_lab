# -*- coding: utf-8 -*-
# @time      : 2018/10/31 16:48
# @author    : rpyxqi@gmail.com
# @file      : s_buy_and_hold.py

from rqalpha.api import *
import time
from quant_models.utils.helper import get_config

config = get_config()


def init(context):
    print('call init')
    context.g_process_time = 0.0
    context.s1 = "000001.XSHE"


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    print('start handle_bar')
    # order_shares(context.s1, 1000)
    ts = time.time()
    order_target_percent(context.s1, 1.0)
    te = time.time()
    context.g_process_time += (te - ts)
    print('end handle_bar')
    print(context.g_process_time)


def after_trading(context):
    pass


__config__ = {
    "base": {
        "start_date": config["feature_mining_strategy"]["start_date"],
        "end_date": config["feature_mining_strategy"]["end_date"],
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": "000300.XSHG",
        "commission-multiplier": 1,
        "margin_multiplier": 1,
        "accounts": {
            "stock": 10000
        }
    },
    "extra": {
        "log_level": "error",
        "show": True,
    },
    "mod": {
        "sys_progress": {
            "enabled": False,
            "show": True,
        },
        "sys_analyser": {
            "enabled": True,
            "show": True,
            "plot": False,
            "output_file": "s_buy_and_hold.pkl",
            "plot": True,
            "plot_save_file": 's_buy_and_hold.png',
        },
        "sys_simulation": {
            "enabled": True,
            "priority": 100,
            "slippage": 0.02,
            "commission_multiplier": 0.0008,
        },
        "sys_risk": {
            "enabled": True,
            "validate_position": False,
        }
    }
}
