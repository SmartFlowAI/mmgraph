# mmgraph

use torchview to visualize the openmmlab 2.0 model

400+ mmyolo mmdetection models has been visualized, mmrotate、mmclassification models are coming soon.

![](https://raw.githubusercontent.com/vansin/mmgraph/a274a417f8ae5d7d2e6d34f14716edd94fcf88ba/mmrotate/configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py.svg)



if you want to visualize your model, you can use the model_visual.ipynb.



```python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config
from functools import partial

# if you want 
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules
register_all_modules()

# from mmdet.registry import MODELS
# from mmdet.utils import register_all_modules
# register_all_modules()
import graphviz


from mmengine.runner import Runner
from torchview import draw_graph
from torchinfo import summary

graphviz.set_default_format('svg')


config = '../mmrotate/configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py'
graph_path = config.replace('mmrotate','model_visual/mmrotate')

cfg = Config.fromfile(config)

dataloader = Runner.build_dataloader(cfg.val_dataloader)

for idx, data_batch in enumerate(dataloader):
    print(idx, data_batch)
    break

model = MODELS.build(cfg.model)
_forward = model.forward

data = model.data_preprocessor(data_batch)
model.forward = partial(_forward, data_samples=data['data_samples'])


summary(model, data['inputs'].shape, depth=3)
# summary(model, (1, 3, 1024, 1024), depth=3)
model_graph = draw_graph(model, input_size=data['inputs'].shape)
model_graph.visual_graph

# model_graph.visual_graph.render(filename=graph_path, view=False, cleanup=True)
```