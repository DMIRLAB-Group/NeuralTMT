import pickle

import torch
from torch.nn.utils.rnn import pad_sequence


def softmax(x, list):
    sum = 0.0
    for i in list:
        sum += torch.exp(i)
    s = torch.exp(x) / (1.0 + sum)
    return s

#load dataset
def load_data_from_dir(dataset):
    if dataset == "nyc":
        dataset_filename = '../data/nyc/nyc_for_our.pkl'
        user_set = {i for i in range(1083)}
        item_set = {i for i in range(9989)}

    elif dataset == "tky":
        dataset_filename = '../data/tky/tky_for_our.pkl'
        user_set = {i for i in range(2293)}
        item_set = {i for i in range(15177)}

    elif dataset == "gowalla":
        dataset_filename = '../data/gowalla/gowalla_for_our.pkl'
        user_set = {i for i in range(1186)}
        item_set = {i for i in range(41275)}

    elif dataset == "weeplace":
        dataset_filename = './data/weeplace/weeplace_for_our.pkl'
        user_set = {i for i in range(13819)}
        item_set = {i for i in range(37471)}

    f = open(dataset_filename, 'rb')  # 保存在项目地址下
    data_list = pickle.load(f)
    f.close()
    return data_list, user_set, item_set

#generate trainning sample
def GenerateTrainSample(tr_data, num_items):
    train_data_list = []
    for sample in tr_data:
        next_day = list(sample[-4:])
        last_day = sample[:-4]
        next_day = pad_sequence([torch.tensor(i) for i in next_day], batch_first=True,
                                padding_value=num_items)
        next_day = next_day.numpy().tolist()
        sample = last_day + tuple(next_day)
        for i in range(len(next_day[0])):
            user_tuple_pre = list(sample)
            temp = [next_day[0][i], next_day[1][i], next_day[2][i], next_day[3][i]]
            user_tuple_pre.extend(temp)
            train_data_list.append(tuple(user_tuple_pre))
    return train_data_list


def set_random_seed(seed=2021):
    # 是的程序无论执行多少次 最终的随机化结果都是相同的
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass
