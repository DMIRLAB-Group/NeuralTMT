import os
import pickle
import random

import torch
from torch.utils.data import Dataset


class BPR_Dataset(Dataset):
    def __init__(self, train_tuples, n_users, n_items, item_set, num_ng, is_training=True, args=None):
        print("=" * 10, "Creating Dataset Object", "=" * 10)
        self.train_tuples = train_tuples
        self.n_users = n_users
        self.n_items = n_items
        self.item_set = item_set
        self.num_ng = num_ng
        self.is_training = is_training
        if self.is_training:
            if os.path.exists("data/" + args.dataset + "/" + args.dataset + '.pkl'):
                pickle_in = open("data/" + args.dataset + "/" + args.dataset + ".pkl", "rb")
                self.new_train_tuples = pickle.load(pickle_in)
            else:
                self.new_train_tuples = self.ng_sample()
                pickle_out = open("data/" + args.dataset + "/" + args.dataset + ".pkl", "wb")
                pickle.dump(self.new_train_tuples, pickle_out)
                pickle_out.close()
        else:
            self.new_train_tuples = self.train_tuples
        self.n_tuples = len(self.new_train_tuples)

    def __len__(self):
        return self.n_tuples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_training:
            uid, iid_mor_pre, iid_aft_pre, iid_eve_pre, iid_deep_pre, \
            iid_mor_next_list, \
            iid_aft_next_list, iid_eve_next_list, iid_deep_next_list, iid_mor_next, iid_aft_next, iid_eve_next, \
            iid_deep_next, iid_mor_negative, iid_aft_negative, \
            iid_eve_negative, iid_deep_negative = self.new_train_tuples[idx]

            sample = (uid, iid_mor_pre, iid_aft_pre, iid_eve_pre, iid_deep_pre,
                      iid_mor_next, iid_aft_next, iid_eve_next, iid_deep_next, iid_mor_next_list, iid_aft_next_list,
                      iid_eve_next_list, iid_deep_next_list, iid_mor_negative,
                      iid_aft_negative, iid_eve_negative, iid_deep_negative)
        else:
            uid, iid_mor_pre, iid_aft_pre, iid_eve_pre, iid_deep_pre, iid_mor_next_list, \
            iid_aft_next_list, iid_eve_next_list, iid_deep_next_list = self.new_train_tuples[idx]

            sample = (uid, iid_mor_pre, iid_aft_pre, iid_eve_pre, iid_deep_pre,
                      iid_mor_next_list, iid_aft_next_list, iid_eve_next_list, iid_deep_next_list)

        return sample

    #ramdom get negative samples
    def ng_sample(self):
        ng_samples = []
        fill_index = max(self.item_set) + 1
        for x in self.train_tuples:
            if x[-4] == fill_index:
                exclu_mor_set = set([x[-4]])
            else:
                exclu_mor_set = self.item_set - set(x[-8])

            if x[-3] == fill_index:
                exclu_aft_set = set([x[-3]])
            else:
                exclu_aft_set = self.item_set - set(x[-7])
            if x[-2] == fill_index:
                exclu_eve_set = set([x[-2]])
            else:
                exclu_eve_set = self.item_set - set(x[-6])

            if x[-1] == fill_index:
                exclu_deep_set = set([x[-1]])
            else:
                exclu_deep_set = self.item_set - set(x[-5])
            #sample num_ng negs
            for t in range(self.num_ng):
                old = x
                mor_neg = random.sample(exclu_mor_set, 1)
                aft_neg = random.sample(exclu_aft_set, 1)
                eve_neg = random.sample(exclu_eve_set, 1)
                deep_neg = random.sample(exclu_deep_set, 1)
                #confirm unique negative samples
                x = tuple(x) + (mor_neg[0], aft_neg[0], eve_neg[0], deep_neg[0])
                sample = x
                x = old
                ng_samples.append(sample)
        return ng_samples
