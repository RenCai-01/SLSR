import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# 设置全局随机种子
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# import numpy as np
# np.random.seed(42)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper functions for abs weight pruning
def sorted_mat(matrix):
    temp = list(abs(matrix).flatten())
    temp.sort()
    return temp


def prune(matrix, mat_sort, to_prune):
    if to_prune != 0:
        alpha = mat_sort[int(to_prune * 0.1 * len(mat_sort))]
        matrix[abs(matrix) <= alpha] = 0
    return matrix


def rank(matrix):
    np_matrix = np.array(matrix)
    return np.linalg.matrix_rank(np_matrix)/min(list(np_matrix.shape))


# What percentage can be pruned by weight
def sparsity(matrix, alpha):
    abs_matrix = abs(matrix)
    filtered_matrix = abs_matrix[abs_matrix < alpha]
    return len(filtered_matrix)/matrix.size


def viz_rank_change(rank_list,name):
    fig = plt.figure()
    plt.plot(rank_list)
    plt.savefig(name)


# Helper functions for rank reduction
# def do_low_rank(weight, k, debug=False, niter=2):
# def do_low_rank(weight, k, model, debug=False, niter=2):
def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    # model_edit_new=deepcopy(model)

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)

    # print(results[0])
    # import time
    # print(time.sleep(2))
    # print('-------------type(results[0])-----------------')
    # print(type(results[0]))
    # print(results[0].shape)
    # print(time.sleep(5))



    # print(results)
    # print(time.sleep(2))
    # print('-------------type(results)-----------------')
    # print(type(results))
    # print(results.shape)
    # print(time.sleep(5))

    # U, S, V = torch.svd_lowrank(weight,
    #                             q=desired_rank,
    #                             niter=niter)  # 2024年10月24日23:34:34 任加的

    # results = torch.svd_lowrank(weight.to(device),
    #                             q=desired_rank)

    U, S, V =results[0].clone(),results[1].clone(),results[2].clone()# 2024年11月1日23:52:11 任加的

# layer_name_weight = "model.layers.26.mlp.gate_proj.weight"  # 假设我们要对第26层的 mlp.gate_proj 权重进行低秩近似
    # # layer_name = "model.layers.26.mlp.gate_proj.weight"
    # layer_name = "model.layers.26.mlp.gate_proj"  # 假设我们要对第26层的 mlp.gate_proj 权重进行低秩近似
    # # 创建新的权重层名称
    # new_layer_name_U = f"{layer_name}_U.weight"
    # new_layer_name_S = f"{layer_name}_S.weight"
    # new_layer_name_V = f"{layer_name}_V.weight"
    #
    # # 将新的权重层添加到模型的状态字典中
    # state_dict = model_edit_new.state_dict()
    # state_dict[new_layer_name_U] = U
    # state_dict[new_layer_name_S] = S
    # state_dict[new_layer_name_V] = V

    # 删除原始权重层
    # del state_dict[layer_name]
    # del state_dict[layer_name_weight]

    # 更新模型的状态字典
    # model_edit_new.load_state_dict(state_dict)





    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    # return weight_approx, model_edit_new, U, S, V  # 2024年10月29日15:04:10 加了USV
    return weight_approx, U, S, V  # 2024年10月29日15:04:10 加了USV
    # return model


import torch
import random
import numpy as np

def add_noise(weight, k=None, debug=False):
    assert weight.ndim == 2

    # 固定随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if debug:
        print(f"Shape is {weight.shape} and dtype is {weight.dtype}")

    # 生成与 weight 形状相同的随机噪声，并加到原矩阵上
    noise = torch.randn_like(weight) * 0.01  # 假设噪声乘一个缩放因子，比如 0.01
    weight_approx = weight + noise

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape == weight.shape

    # 转换为 Parameter
    weight_approx = torch.nn.Parameter(weight_approx)

    # 由于不再使用 SVD，U、S、V 设为 None
    # return weight_approx, None, None, None
    return weight_approx
