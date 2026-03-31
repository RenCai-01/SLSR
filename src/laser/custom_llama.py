#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 23:24
# @Author  : Castrol
# @Site    : 
# @File    : custom_llama.py
# @Software: PyCharm
# @Function: 
# @Attention:
# 自己加的
import os
import json
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from copy import deepcopy

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义一个函数来进行 SVD 替换
def replace_gate_proj_with_svd_lowrank(layer, U, S, Vt, q=None, device=device):
    if U is None or S is None or Vt is None:
        raise ValueError("U, S, and Vt must be provided for SVD replacement")

    print("Performing SVD replacement with shapes:")
    print(f"U shape: {U.shape}")
    print(f"S shape: {S.shape}")
    print(f"Vt shape: {Vt.shape}")

    # 设置随机种子以确保每次运行时结果一致
    torch.manual_seed(42)

    # 定义新的三层结构
    fc1 = nn.Linear(layer.in_features, S.size(0), bias=layer.bias is not None).to(device)
    fc2 = nn.Linear(S.size(0), S.size(0), bias=False).to(device)
    fc3 = nn.Linear(S.size(0), layer.out_features, bias=layer.bias is not None).to(device)

    # 设置权重
    fc1.weight.data = U.to(device)
    fc2.weight.data = torch.diag(S).to(device)
    fc3.weight.data = Vt.T.to(device)  # 注意这里需要转置，因为 svd_lowrank 返回的是 V 而不是 Vt

    # 设置偏置项
    if layer.bias is not None:
        fc3.bias.data = layer.bias.to(device)

    # 返回新的序列层
    return nn.Sequential(fc1, fc2, fc3).to(device)

# 创建自定义模型类
class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, U=None, S=None, Vt=None, layer_index=None,rank=0, device=device):
        super().__init__(config)
        print("Model initialized.")

        # 如果提供了U, S, Vt，则应用SVD修改
        if U is not None and S is not None and Vt is not None:
            print("Applying SVD modification...")
            self.apply_svd_modification(U, S, Vt,layer_index, device=device)

    def apply_svd_modification(self, U, S, Vt, layer_index, device=device):
        if U is None or S is None or Vt is None:
            raise ValueError("U, S, and Vt must be provided for SVD replacement")

        print(f"Applying SVD modification to layer {layer_index}...")
        gate_proj_layer = self.model.layers[layer_index].mlp.gate_proj

        # 检查是否已经是一个 Sequential 对象
        if isinstance(gate_proj_layer, nn.Sequential):
            print(f"Layer {layer_index} is already modified, skipping.")
            return

        new_gate_proj = replace_gate_proj_with_svd_lowrank(gate_proj_layer, U, S, Vt, q=None, device=device)
        self.model.layers[layer_index].mlp.gate_proj = new_gate_proj

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)

        config.svd_modified=True  #2024年11月15日23:43:50 加的
        # 检查配置中的标记字段
        if config.svd_modified:
            # 先用默认方式加载模型
            model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            # 读取索引文件
            index_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin.index.json')
            with open(index_file, 'r') as f:
                index = json.load(f)

            # 加载所有分片文件
            state_dict = {}
            for shard_file in index['weight_map'].values():
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                shard_state_dict = torch.load(shard_path, map_location='cpu')  # 先加载到 CPU
                state_dict.update(shard_state_dict)

            # 获取U, S, Vt参数
            layer_indices = kwargs.get('layer_indices', [0, 1])  # 默认修改第0,1层
            U = {layer_index: state_dict.get(f'model.layers.{layer_index}.mlp.gate_proj.0.weight') for layer_index in layer_indices}
            S = {layer_index: state_dict.get(f'model.layers.{layer_index}.mlp.gate_proj.1.weight') for layer_index in layer_indices}
            Vt = {layer_index: state_dict.get(f'model.layers.{layer_index}.mlp.gate_proj.2.weight') for layer_index in layer_indices}

            # 如果U, S, Vt存在，则应用SVD修改
            if all(u is not None and s is not None and vt is not None for u, s, vt in zip(U.values(), S.values(), Vt.values())):
                # 将模型转换为自定义模型类
                custom_model = cls(config, device=device)
                for layer_index in layer_indices:
                    custom_model.apply_svd_modification(U[layer_index], S[layer_index], Vt[layer_index], layer_index, device=device)
                return custom_model
            else:
                raise ValueError("U, S, and Vt must be provided for SVD replacement")
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model


