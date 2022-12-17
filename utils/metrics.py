import numpy as np


def dcg(x, y, k):
    x = x[:k]
    x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)
    cg = x_in_y / np.log2(2 + np.arange(len(x)))  # cumulative gain at every position
    _dcg = cg.sum()
    return _dcg


def ndcg(x, y, k):
    dcg_k = dcg(x, y, k)
    idcg_k = dcg(y, y, k)
    ndcg_k = dcg_k / idcg_k
    return ndcg_k


def hr(scores_top, iid_pos_list):
    hit_list = np.isin(scores_top, iid_pos_list, assume_unique=True).astype(np.int)
    return hit_list.sum()


def recall(scores_top, iid_pos_list, k, normalize=True):
    hit_list = np.isin(scores_top, iid_pos_list, assume_unique=True).astype(np.int)
    normalization = min(k, len(iid_pos_list)) if normalize else len(iid_pos_list)
    return hit_list.sum() / normalization
