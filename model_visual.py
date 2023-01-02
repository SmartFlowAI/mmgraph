# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config
from functools import partial
import os

# if you want 
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules
register_all_modules()

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
register_all_modules()

from mmyolo.utils import register_all_modules
register_all_modules()


from mmengine.runner import Runner

import graphviz
from torchview import draw_graph
from torchinfo import summary
# graphviz.set_jupyter_format('svg')

# graphviz.set_default_engine('')
graphviz.set_default_format('svg')


import pandas as pd


def draw_model(config, filename):

    cfg = Config.fromfile(config)

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    for idx, data_batch in enumerate(dataloader):
        # print(idx, data_batch)
        break

    model = MODELS.build(cfg.model)
    _forward = model.forward

    data = model.data_preprocessor(data_batch)
    model.forward = partial(_forward, data_samples=data['data_samples'])


    # summary(model, data['inputs'].shape, depth=3)
    # summary(model, (1, 3, 1024, 1024), depth=3)
    model_graph = draw_graph(model, input_size=data['inputs'].shape)
    model_graph.visual_graph

    model_graph.visual_graph.render(filename=filename, view=False, cleanup=True)



result_configs = []


if __name__ == '__main__':

    # prefix = 'mmdetection/configs/'
    prefix = '../mmyolo/configs/'

    algorithms = os.listdir(prefix)

    configs_list = []
    for algorithm in algorithms:
        configs = os.listdir(prefix + algorithm)
        if 'metafile.yml' in configs:
            for config in configs:
                if config.endswith('.py'):
                    configs_list.append(prefix + algorithm + '/' + config)

    # print(configs_list)


    for i, config in enumerate(configs_list):
        print(i, config, configs_list.__len__())
        
        
        result_dict = dict()
        result_dict['config'] = config

        try:
            # draw_model(config, config.replace('mmyolo', 'model_visual'))
            draw_model(config, config.replace('mmyolo', 'model_visual/mmyolo'))
            result_dict['result'] = 'PASS'
            print('PASS')
        except Exception as e:
            result_dict['result'] = 'FAIL'
            result_dict['error'] = str(e)
            print(e)
            print('FAIL')
            # print(config)
    
        result_configs.append(result_dict)


    pd.DataFrame(result_configs).to_csv('result.csv')