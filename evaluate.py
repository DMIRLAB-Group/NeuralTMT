import numpy as np
from tqdm import tqdm

from metrics import ndcg, hr, recall


def evaluate_model(model, test_loader, is_training=None, item_num=None, logger=None):
    model.eval()
    num_1, num_5, num_10 = 0, 0, 0
    ndcg_num = 0
    HR_1 = 0
    HR_5 = 0
    HR_10 = 0
    NDCG_1 = 0
    NDCG_5 = 0
    NDCG_10 = 0

    for idx, sample in enumerate(tqdm(test_loader)):
        if is_training:
            uid, basket_1, basket_2, basket_3, basket_4, iid_1, iid_2, iid_3, iid_4, iid_mor_next_list, \
            iid_aft_next_list, iid_eve_next_list, iid_deep_next_list, neg_iid_1, neg_iid_2, \
            neg_iid_3, neg_iid_4 = sample
        else:
            uid, basket_1, basket_2, basket_3, basket_4, \
            iid_mor_next_list, iid_aft_next_list, iid_eve_next_list, iid_deep_next_list = sample
        scores_1, scores_2, scores_3, scores_4 = model.compute_transpro_batch(uid, basket_1,
                                                                              basket_2, basket_3, basket_4)
        scores_1 = scores_1.cpu().numpy()
        scores_2 = scores_2.cpu().numpy()
        scores_3 = scores_3.cpu().numpy()
        scores_4 = scores_4.cpu().numpy()

        for i in range(len(uid)):
            scores1_top1 = scores_1[i][:1]
            scores1_top5 = scores_1[i][:5]
            scores1_top10 = scores_1[i][:10]

            scores2_top1 = scores_2[i][:1]
            scores2_top5 = scores_2[i][:5]
            scores2_top10 = scores_2[i][:10]

            scores3_top1 = scores_3[i][:1]
            scores3_top5 = scores_3[i][:5]
            scores3_top10 = scores_3[i][:10]

            scores4_top1 = scores_4[i][:1]
            scores4_top5 = scores_4[i][:5]
            scores4_top10 = scores_4[i][:10]

            # top1
            hit_mor_1 = hr(scores1_top1, iid_mor_next_list[i])
            hit_aft_1 = hr(scores2_top1, iid_aft_next_list[i])
            hit_eve_1 = hr(scores3_top1, iid_eve_next_list[i])
            hit_deep_1 = hr(scores4_top1, iid_deep_next_list[i])

            # top5
            hit_mor_5 = hr(scores1_top5, iid_mor_next_list[i])
            hit_aft_5 = hr(scores2_top5, iid_aft_next_list[i])
            hit_eve_5 = hr(scores3_top5, iid_eve_next_list[i])
            hit_deep_5 = hr(scores4_top5, iid_deep_next_list[i])

            #top10
            hit_mor_10 = hr(scores1_top10, iid_mor_next_list[i])
            hit_aft_10 = hr(scores2_top10, iid_aft_next_list[i])
            hit_eve_10 = hr(scores3_top10, iid_eve_next_list[i])
            hit_deep_10 = hr(scores4_top10, iid_deep_next_list[i])

            HR_1 += hit_mor_1 + hit_aft_1 + hit_eve_1 + hit_deep_1
            HR_5 += hit_mor_5 + hit_aft_5 + hit_eve_5 + hit_deep_5
            HR_10 += hit_mor_10 + hit_aft_10 + hit_eve_10 + hit_deep_10

            if iid_mor_next_list[i] != [item_num]:
                NDCG_1 += ndcg(scores1_top1, iid_mor_next_list[i], 1)
                NDCG_5 += ndcg(scores1_top5, iid_mor_next_list[i], 5)
                NDCG_10 += ndcg(scores1_top10, iid_mor_next_list[i], 10)
                ndcg_num += 1
                num_1 += min(1, len(iid_mor_next_list[i]))
                num_5 += min(5, len(iid_mor_next_list[i]))
                num_10 += min(10, len(iid_mor_next_list[i]))


            if iid_aft_next_list[i] != [item_num]:
                NDCG_1 += ndcg(scores2_top1, iid_aft_next_list[i], 1)
                NDCG_5 += ndcg(scores2_top5, iid_aft_next_list[i], 5)
                NDCG_10 += ndcg(scores2_top10, iid_aft_next_list[i], 10)

                ndcg_num += 1
                num_1 += min(1, len(iid_aft_next_list[i]))
                num_5 += min(5, len(iid_aft_next_list[i]))
                num_10 += min(10, len(iid_aft_next_list[i]))

            if iid_eve_next_list[i] != [item_num]:
                NDCG_1 += ndcg(scores3_top1, iid_eve_next_list[i], 1)
                NDCG_5 += ndcg(scores3_top5, iid_eve_next_list[i], 5)
                NDCG_10 += ndcg(scores3_top10, iid_eve_next_list[i], 10)
                ndcg_num += 1
                num_1 += min(1, len(iid_eve_next_list[i]))
                num_5 += min(5, len(iid_eve_next_list[i]))
                num_10 += min(10, len(iid_eve_next_list[i]))

            if iid_deep_next_list[i] != [item_num]:
                NDCG_1 += ndcg(scores4_top1, iid_deep_next_list[i], 1)
                NDCG_5 += ndcg(scores4_top5, iid_deep_next_list[i], 5)
                NDCG_10 += ndcg(scores4_top10, iid_deep_next_list[i], 10)
                ndcg_num += 1
                num_1 += min(1, len(iid_deep_next_list[i]))
                num_5 += min(5, len(iid_deep_next_list[i]))
                num_10 += min(10, len(iid_deep_next_list[i]))

    HR_1 /= num_1
    HR_5 /= num_5
    HR_10 /= num_10
    NDCG_1 /= ndcg_num
    NDCG_5 /= ndcg_num
    NDCG_10 /= ndcg_num

    logger.info('HR@{0}={1}'.format(1, HR_1))
    logger.info('HR@{0}={1}'.format(5, HR_5))
    logger.info('HR@{0}={1}'.format(10, HR_10))
    logger.info('NDCG@{0}={1}'.format(1, NDCG_1))
    logger.info('NDCG@{0}={1}'.format(5, NDCG_5))
    logger.info('NDCG@{0}={1}'.format(10, NDCG_10))
    return HR_5, HR_10, NDCG_5, NDCG_10
