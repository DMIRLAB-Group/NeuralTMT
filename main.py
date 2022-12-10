import argparse
import collections.abc as container_abcs
import os

from sklearn.model_selection import train_test_split

from evaluate import evaluate_model
from model.NeuralTMT import NeuralTMT

int_classes = int
string_classes = str
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import re
from bpr_dataset import BPR_Dataset
from bpr_loss import BPR_Loss
import torch.utils.data as data
from load_data import load_data_from_dir, set_random_seed, GenerateTrainSample
from logger import Logger


def rewrite_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    global _use_shared_memory
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: rewrite_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], list):
        return batch
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [rewrite_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def train():
    best_hr_5 = 0
    best_hr_10 = 0
    best_ndcg_5 = 0
    best_ndcg_10 = 0
    for epoch in range(1, args.n_epoch + 1):
        All_loss = 0.0
        model.train()
        logger.info('{0} training start：'.format(epoch))
        for idx, sample in enumerate(tqdm(train_loader)):
            uid, basket_1, basket_2, basket_3, basket_4, iid_1, iid_2, iid_3, iid_4, iid_mor_next_list, \
            iid_aft_next_list, iid_eve_next_list, iid_deep_next_list, neg_iid_1, neg_iid_2, \
            neg_iid_3, neg_iid_4 = sample
            uid = uid.to(device=model.device)

            #padding
            basket_1 = pad_sequence([torch.tensor(i) for i in basket_1], batch_first=True,
                                    padding_value=max(item_set) + 1)
            basket_2 = pad_sequence([torch.tensor(i) for i in basket_2], batch_first=True,
                                    padding_value=max(item_set) + 1)
            basket_3 = pad_sequence([torch.tensor(i) for i in basket_3], batch_first=True,
                                    padding_value=max(item_set) + 1)
            basket_4 = pad_sequence([torch.tensor(i) for i in basket_4], batch_first=True,
                                    padding_value=max(item_set) + 1)

            basket_1 = basket_1.to(device=model.device)
            basket_2 = basket_2.to(device=model.device)
            basket_3 = basket_3.to(device=model.device)
            basket_4 = basket_4.to(device=model.device)
            iid_1 = iid_1.to(device=model.device)
            iid_2 = iid_2.to(device=model.device)
            iid_3 = iid_3.to(device=model.device)
            iid_4 = iid_4.to(device=model.device)
            neg_iid_1 = neg_iid_1.to(device=model.device)
            neg_iid_2 = neg_iid_2.to(device=model.device)
            neg_iid_3 = neg_iid_3.to(device=model.device)
            neg_iid_4 = neg_iid_4.to(device=model.device)

            # forward + backward + optimize
            pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3, pos_prob_4, neg_prob_4 \
                = model(uid, basket_1, basket_2, basket_3, basket_4, iid_1, iid_2,
                        iid_3, iid_4, neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4)

            optimizer.zero_grad()
            loss = criterion(pos_prob_1, neg_prob_1
                             , pos_prob_2, neg_prob_2, pos_prob_3,
                             neg_prob_3, pos_prob_4, neg_prob_4, optimizer, args)

            # regular
            # for param in model.IL_1.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_1.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.IL_2.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_2.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.IL_3.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_3.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.IL_4.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_4.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_4.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.LI_4.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.UI_1.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.UI_2.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.UI_3.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.UI_4.parameters(): loss += args.regular * torch.norm(param)
            # for param in model.IU_1.parameters(): loss += args.regular * torch.norm(param)
            All_loss += loss.item()
            loss.backward()
            optimizer.step()
        # eval
        logger.info('{0} training end，loss：{1}'.format(epoch, All_loss / (len(train_loader) * args.batch_size * 4)))

        HR_5, HR_10, NDCG_5, NDCG_10 = evaluate_model(model, test_loader, is_training=False, item_num=item_num,
                                                      logger=logger)
        if (HR_5 > best_hr_5) | (HR_10 > best_hr_10) | (NDCG_5 > best_ndcg_5) | (NDCG_10 > best_ndcg_10):
            better = 0
            if HR_5 > best_hr_5: better += 1
            if HR_10 > best_hr_10: better += 1
            if NDCG_5 > best_ndcg_5: better += 1
            if NDCG_10 > best_ndcg_10: better += 1
            if better > 1:
                model_path = os.path.join('./data/' + args.dataset + '/' + args.dataset + '_model',
                                          '{0}_dim{1}_neg{2}_attention.pt'.
                                          format(args.dataset, args.n_factor, args.n_neg))
                dir_name = os.path.dirname(model_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                torch.save(model, model_path)
                logger.info("Save model as %s" % model_path)

            best_hr_5 = max(HR_5, best_hr_5)
            best_hr_10 = max(HR_10, best_hr_10)
            best_ndcg_5 = max(NDCG_5, best_ndcg_5)
            best_ndcg_10 = max(NDCG_10, best_ndcg_10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gowalla')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=50)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=30)
    parser.add_argument('--batch_size', help='# of batch samples', type=int, default=256)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=500)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.00003)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.0001)
    parser.add_argument('-s', '--seed', help='', type=int, default=2021)
    parser.add_argument('--z_init', help='', type=float, default=4.0)
    parser.add_argument('--std_init', help='', type=float, default=1)

    args = parser.parse_args()
    set_random_seed(args.seed)
    logger_path = os.path.join('./data/' + args.dataset + '/' + args.dataset + '_log',
                               'dim{0}_neg{1}_std_init{2}_z_init{3}_regular{4}_batch{5}.log'.
                               format(args.n_factor, args.n_neg, args.std_init, args.z_init,
                                      args.regular, args.batch_size))
    logger = Logger(logger_path)

    logger.info('--dataset:{0}'.format(args.dataset))
    logger.info('--batch_size:{0}:'.format(args.batch_size))
    logger.info('--learn_rate:{0}'.format(args.learn_rate))
    logger.info('--n_factor:{0}'.format(args.n_factor))
    logger.info('--ng:{0}'.format(args.n_neg))

    data_list, user_set, item_set = load_data_from_dir(args.dataset)
    tr_data, te_data = train_test_split(data_list, test_size=0.2, random_state=args.seed)
    item_num = len(item_set)
    # generate trainning sample
    tr_data = GenerateTrainSample(tr_data, item_num)

    bpr_traindata = BPR_Dataset(tr_data, max(user_set) + 1, max(item_set) + 1, item_set,
                                args.n_neg, is_training=True, args=args)
    train_loader = data.DataLoader(bpr_traindata,
                                   batch_size=args.batch_size, shuffle=True,
                                   collate_fn=rewrite_collate, num_workers=1)
    bpr_testdata = BPR_Dataset(te_data, max(user_set) + 1, max(item_set) + 1, item_set,
                               args.n_neg, is_training=False, args=args)
    test_loader = data.DataLoader(bpr_testdata, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=rewrite_collate, num_workers=1)
    model = NeuralTMT(n_users=max(user_set) + 1, n_items=max(item_set) + 1, k_UI=args.n_factor, k_IL=args.n_factor,
                      z_m=args.z_init)
    model.to(device=model.device)
    criterion = BPR_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    train()
