import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_opponent():
    # BEGIN YOUR CODE
    from submission import GobangModel
    import os

    opponent = GobangModel(board_size=board_size, bound=bound)

    # 尝试加载对手模型权重
    # 对手模型通常是固定的基准模型，或者之前的某个强力版本
    opponent_path = './checkpoints/model_2999.pth'

    if os.path.exists(opponent_path):
        try:
            opponent.load_state_dict(torch.load(
                opponent_path, map_location=device))
            print(f"Successfully loaded opponent from {opponent_path}")
        except Exception as e:
            print(f"Failed to load opponent from {opponent_path}: {e}")
            print("Returning initialized opponent without weights.")
    else:
        # 如果找不到特定的 opponent.pth，也可以尝试加载 model.pth 作为对手
        # 这样就是“左右互搏”，自己打自己，符合readme中的naive self play策略
        if os.path.exists('model.pth'):
            print(
                f"Opponent file {opponent_path} not found. Using model.pth as opponent.")
            try:
                opponent.load_state_dict(torch.load(
                    'model.pth', map_location=device))
            except Exception as e:
                print(f"Failed to load model.pth as opponent: {e}")
        else:
            print(f"No opponent file found. Returning initialized opponent.")

    opponent.to(device)
    return opponent
    # END YOUR CODE


__all__ = ['get_opponent']
