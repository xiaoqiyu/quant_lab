# -*- coding: utf-8 -*-
# @time      : 2018/12/10 15:58
# @author    : rpyxqi@gmail.com
# @file      : io_utils.py

import json
import os
from ..utils.helper import get_source_root


def write_json_file(file_path='', data=None):
    if not data:
        return
    with open(file_path, 'w') as outfile:
        j_data = json.dumps(data)
        outfile.write(j_data)


def load_json_file(filepath=''):
    with open(filepath) as infile:
        contents = infile.read()
        return json.loads(contents)


if __name__ == '__main__':
    features = {'growth': [],
                'vs': [],
                'return': [],
                'ma': [],
                'obos': [],
                'power': [],
                'trend': [],
                'volume': [],
                'psi': [],
                'pq': [],
                'sc': [],
                'cf': [],
                'oc': [],
                'af': [],
                'derive': [],
                }
    # write_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\features.json', features)
    # print(load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\features.json'))
    # write_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\selected_stocks.json', {'20181203':[]})
    ret = load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\selected_stocks.json')
    dates = []
    adj_map = {}
    import pprint

    pprint.pprint(ret)
    # for d, l in ret.items():
    #
    #     if l:
    #         # dates.append((d, len(l)))
    #         adj_map.update({d[:-2]: l})
    # # write_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\selected_stocks.json', adj_map)
    # import pprint
    # pprint.pprint(adj_map)
    root = get_source_root()
    feature_mapping = os.path.join(os.path.realpath(root), 'conf', 'feature_style.json')
    write_json_file('E:\pycharm\quant\quant_models\quant_models\conf\\selected_stocks.json', adj_map)
