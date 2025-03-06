import numpy as np

# 将连续的动作映射到离散的动作向量中
# def pad_action(act, act_param):
#     action = np.zeros((7,))
#     action[0] = act
#     if act == 0:
#         action[[1, 2]] = act_param
#     elif act == 1:
#         action[3] = act_param
#     elif act == 2:
#         action[[4, 5]] = act_param
#     elif act == 3:
#         action[[6]] = act_param
#     else:
#         raise ValueError("Unknown action index '{}'".format(act))
#     return action
# num_power_channels = 3
# num_beams = 7


def pad_action(action, action_parameters):
    # result = np.concatenate((data1, data2))

    # 映射到0-100的范围
    min_val = np.min(action_parameters)
    max_val = np.max(action_parameters)
    mapped_data2 = 1 + (action_parameters - min_val) * (499 / (max_val - min_val))

    # 替换连接后的数据
    # result = np.concatenate((action, mapped_data2))

    # return result
    return (action, np.array(mapped_data2, dtype=np.float32))
#
# data1 = np.array([26])
# data2 = np.array([-62.116943, -99.569595, 3.16708, -348.8092])
#
# result = pad_action(data1, data2)
# print(result)




