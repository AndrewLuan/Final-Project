import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_model():
    from submission import GobangModel
    import os
    
    model = GobangModel(board_size=board_size, bound=bound)
    
    # 尝试加载模型权重，如果不存在则返回未训练的模型（或根据需求抛出错误）
    # 这里假设模型保存在当前目录下的 model.pth 或 checkpoints 目录中
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        # 尝试查找 checkpoints 目录下的最新模型
        if os.path.exists('checkpoints'):
            checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if checkpoints:
                # 简单的逻辑：取名字里数字最大的，或者按时间排序
                # 这里假设文件名格式为 model_xxx.pth，取最后生成的
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0)
                model_path = os.path.join('checkpoints', checkpoints[-1])
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Returning initialized model without weights.")
    else:
        print(f"No model file found at {model_path}. Returning initialized model.")

    model.to(device)
    return model


__all__ = ['get_model']
