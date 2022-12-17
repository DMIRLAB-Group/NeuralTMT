import torch


class BPR_Loss(object):
    def __init__(self):
        print("=" * 10, "Creating Model Loss Function", "=" * 10)
        pass

    def __call__(self, pos_scores_1, ne_scores_1, pos_scores_2, ne_scores_2, pos_scores_3, ne_scores_3,
                 pos_scores_4, ne_scores_4, optimizer, args):
        loss_1 = torch.log(1 + torch.exp(-pos_scores_1 + ne_scores_1)).sum()
        loss_2 = torch.log(1 + torch.exp(-pos_scores_2 + ne_scores_2)).sum()
        loss_3 = torch.log(1 + torch.exp(-pos_scores_3 + ne_scores_3)).sum()
        loss_4 = torch.log(1 + torch.exp(-pos_scores_4 + ne_scores_4)).sum()
        return loss_1 + loss_2 + loss_3 + loss_4
