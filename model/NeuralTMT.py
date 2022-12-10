import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class NeuralTMT(nn.Module):
    def __init__(self, n_users, n_items, k_UI=None, k_IL=None, z_m=None):
        super(NeuralTMT, self).__init__()
        print("=" * 10, "Creating our Model-NeuralTMT", "=" * 10)
        # set gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # GPU训练
        self.n_users = n_users
        self.n_items = n_items
        self.k_UI = k_UI
        self.k_IL = k_IL

        # ~V_i^m
        self.IL_1 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.IL_2 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.IL_3 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.IL_4 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)

        # V_i^m
        self.LI_1 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.LI_2 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.LI_3 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)
        self.LI_4 = nn.Embedding(self.n_items + 1, self.k_IL, padding_idx=-1)

        # ~V_u^m
        self.UI_1 = nn.Embedding(self.n_users, self.k_UI)
        self.UI_2 = nn.Embedding(self.n_users, self.k_UI)
        self.UI_3 = nn.Embedding(self.n_users, self.k_UI)
        self.UI_4 = nn.Embedding(self.n_users, self.k_UI)

        # ~V_i
        self.IU = nn.Embedding(self.n_items + 1, self.k_UI, padding_idx=-1)

        # z_m
        self.alpha_mor = nn.Parameter(torch.tensor(z_m))
        self.alpha_aft = nn.Parameter(torch.tensor(z_m))
        self.alpha_eve = nn.Parameter(torch.tensor(z_m))
        self.alpha_deep = nn.Parameter(torch.tensor(z_m))

        # # the parameters for attention layer
        # self.W_mor = torch.nn.Linear(self.k_IL, self.k_IL)
        # self.W_aft = torch.nn.Linear(self.k_IL, self.k_IL)
        # self.W_eve = torch.nn.Linear(self.k_IL, self.k_IL)
        # self.W_deep = torch.nn.Linear(self.k_IL, self.k_IL)

        self.init_model()

    def init_model(self):
        with torch.no_grad():
            self.IL_1.weight.data.normal_(0, 0.01)
            self.IL_2.weight.data.normal_(0, 0.01)
            self.IL_3.weight.data.normal_(0, 0.01)
            self.IL_4.weight.data.normal_(0, 0.01)
            self.IL_1.weight[-1].fill_(0)
            self.IL_2.weight[-1].fill_(0)
            self.IL_3.weight[-1].fill_(0)
            self.IL_4.weight[-1].fill_(0)
            self.LI_1.weight.data.normal_(0, 1)
            self.LI_2.weight.data.normal_(0, 1)
            self.LI_3.weight.data.normal_(0, 1)
            self.LI_4.weight.data.normal_(0, 1)

            self.UI_1.weight.data.normal_(0, 1)
            self.UI_2.weight.data.normal_(0, 1)
            self.UI_3.weight.data.normal_(0, 1)
            self.UI_4.weight.data.normal_(0, 1)

            self.IU.weight.data.normal_(0, 0.01)

            self.LI_1.weight[-1].fill_(0)
            self.LI_2.weight[-1].fill_(0)
            self.LI_3.weight[-1].fill_(0)
            self.LI_4.weight[-1].fill_(0)
            self.IU.weight[-1].fill_(0)
            # nn.init.xavier_uniform_(self.W_mor.weight)
            # nn.init.xavier_uniform_(self.W_aft.weight)
            # nn.init.xavier_uniform_(self.W_eve.weight)
            # nn.init.xavier_uniform_(self.W_deep.weight)
            #
            # nn.init.xavier_uniform_(self.gate_mor.weight)
            # nn.init.xavier_uniform_(self.gate_aft.weight)
            # nn.init.xavier_uniform_(self.gate_eve.weight)
            # nn.init.xavier_uniform_(self.gate_deep.weight)
            #
            # nn.init.xavier_uniform_(self.gate_mor_user.weight)
            # nn.init.xavier_uniform_(self.gate_aft_user.weight)
            # nn.init.xavier_uniform_(self.gate_eve_user.weight)
            # nn.init.xavier_uniform_(self.gate_deep_user.weight)

    def forward(self, uid, basket_1, basket_2, basket_3, basket_4, iid_1, iid_2, iid_3, iid_4,
                neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4):
        pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3 \
            , pos_prob_4, neg_prob_4 = self.predict_period(uid, basket_1, basket_2, basket_3,
                                                         basket_4, iid_1, iid_2, iid_3, iid_4, neg_iid_1,
                                                         neg_iid_2, neg_iid_3, neg_iid_4)

        return pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3 \
            , pos_prob_4, neg_prob_4

    # Aggregation Layer
    def aggregate(self, basket_1, basket_2, basket_3, basket_4):
        # mean pooling
        fmc_1 = torch.mean(self.LI_1(basket_1), dim=1).unsqueeze(1)
        fmc_2 = torch.mean(self.LI_2(basket_2), dim=1).unsqueeze(1)
        fmc_3 = torch.mean(self.LI_3(basket_3), dim=1).unsqueeze(1)
        fmc_4 = torch.mean(self.LI_4(basket_4), dim=1).unsqueeze(1)
        fmc_seq = torch.cat((fmc_1, fmc_2, fmc_3, fmc_4), 1)
        return fmc_seq

    #Attention function
    def Normal_attention(self, target_embedding, input_embedding):
        Q = input_embedding
        weight = torch.matmul(Q, target_embedding.transpose(1, 2))
        # scale
        weight = weight / (Q.shape[-1] ** 0.5)
        # mask
        paddings = (torch.ones(weight.shape) * (-2 ** 32 + 1)).to(weight.device)
        weight = torch.nn.Softmax(dim=1)(torch.where(torch.BoolTensor(weight.cpu() == 0.0).to(weight.device)
                                                     , paddings, weight))
        # weight.data.masked_fill(mask,-2 ** 32 + 1)
        return torch.multiply(input_embedding, weight)

    # Attention Layer
    def encoder_norm_att(self, input_emb, iid_1, iid_2, iid_3, iid_4,
                         neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4):
        fmc_seq_mor = self.Normal_attention(self.IL_1(iid_1).unsqueeze(1), input_emb)
        fmc_seq_mor_neg = self.Normal_attention(self.IL_1(neg_iid_1).unsqueeze(1), input_emb)
        fmc_seq_aft = self.Normal_attention(self.IL_2(iid_2).unsqueeze(1), input_emb)
        fmc_seq_aft_neg = self.Normal_attention(self.IL_2(neg_iid_2).unsqueeze(1), input_emb)
        fmc_seq_eve = self.Normal_attention(self.IL_3(iid_3).unsqueeze(1), input_emb)
        fmc_seq_eve_neg = self.Normal_attention(self.IL_3(neg_iid_3).unsqueeze(1), input_emb)
        fmc_seq_deep = self.Normal_attention(self.IL_4(iid_4).unsqueeze(1), input_emb)
        fmc_seq_deep_neg = self.Normal_attention(self.IL_4(neg_iid_4).unsqueeze(1), input_emb)

        return fmc_seq_mor, fmc_seq_aft, fmc_seq_eve, fmc_seq_deep, fmc_seq_mor_neg, fmc_seq_aft_neg \
            , fmc_seq_eve_neg, fmc_seq_deep_neg

    def fuse(self,uid, output_1, output_2, output_3, output_4, output_1_neg, output_2_neg, output_3_neg, output_4_neg,
             iid_1, iid_2, iid_3, iid_4, neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4):
        u_emb_1 = self.UI_1(uid)
        u_emb_2 = self.UI_2(uid)
        u_emb_3 = self.UI_3(uid)
        u_emb_4 = self.UI_4(uid)

        pos_mor_emb = self.IL_1(iid_1).unsqueeze(-1)
        neg_mor_emb = self.IL_1(neg_iid_1).unsqueeze(-1)

        pos_aft_emb = self.IL_2(iid_2).unsqueeze(-1)
        neg_aft_emb = self.IL_2(neg_iid_2).unsqueeze(-1)

        pos_eve_emb = self.IL_3(iid_3).unsqueeze(-1)
        neg_eve_emb = self.IL_3(neg_iid_3).unsqueeze(-1)

        pos_deep_emb = self.IL_4(iid_4).unsqueeze(-1)
        neg_deep_emb = self.IL_4(neg_iid_4).unsqueeze(-1)

        #long-term
        mf_mor_pos = torch.sum(u_emb_1 * self.IU(iid_1), dim=1)
        mf_mor_neg = torch.sum(u_emb_1 * self.IU(neg_iid_1), dim=1)
        mf_aft_pos = torch.sum(u_emb_2 * self.IU(iid_2), dim=1)
        mf_aft_neg = torch.sum(u_emb_2 * self.IU(neg_iid_2), dim=1)
        mf_eve_pos = torch.sum(u_emb_3 * self.IU(iid_3), dim=1)
        mf_eve_neg = torch.sum(u_emb_3 * self.IU(neg_iid_3), dim=1)
        mf_deep_pos = torch.sum(u_emb_4 * self.IU(iid_4), dim=1)
        mf_deep_neg = torch.sum(u_emb_4 * self.IU(neg_iid_4), dim=1)

        #short-term
        pos_mor = torch.bmm(output_1, pos_mor_emb).sum(1).squeeze(1)
        neg_mor = torch.bmm(output_1_neg, neg_mor_emb).sum(1).squeeze(1)
        pos_aft = torch.bmm(output_2, pos_aft_emb).sum(1).squeeze(1)
        neg_aft = torch.bmm(output_2_neg, neg_aft_emb).sum(1).squeeze(1)
        pos_eve = torch.bmm(output_3, pos_eve_emb).sum(1).squeeze(1)
        neg_eve = torch.bmm(output_3_neg, neg_eve_emb).sum(1).squeeze(1)
        pos_deep = torch.bmm(output_4, pos_deep_emb).sum(1).squeeze(1)
        neg_deep = torch.bmm(output_4_neg, neg_deep_emb).sum(1).squeeze(1)

        pos_prob_1 = torch.sigmoid(self.alpha_mor) * pos_mor + (1 - torch.sigmoid(self.alpha_mor)) * mf_mor_pos
        neg_prob_1 = torch.sigmoid(self.alpha_mor) * neg_mor + (1 - torch.sigmoid(self.alpha_mor)) * mf_mor_neg
        pos_prob_2 = torch.sigmoid(self.alpha_aft) * pos_aft + (1 - torch.sigmoid(self.alpha_aft)) * mf_aft_pos
        neg_prob_2 = torch.sigmoid(self.alpha_aft) * neg_aft + (1 - torch.sigmoid(self.alpha_aft)) * mf_aft_neg
        pos_prob_3 = torch.sigmoid(self.alpha_eve) * pos_eve + (1 - torch.sigmoid(self.alpha_eve)) * mf_eve_pos
        neg_prob_3 = torch.sigmoid(self.alpha_eve) * neg_eve + (1 - torch.sigmoid(self.alpha_eve)) * mf_eve_neg
        pos_prob_4 = torch.sigmoid(self.alpha_deep) * pos_deep + (1 - torch.sigmoid(self.alpha_deep)) * mf_deep_pos
        neg_prob_4 = torch.sigmoid(self.alpha_deep) * neg_deep + (1 - torch.sigmoid(self.alpha_deep)) * mf_deep_neg

        return pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3, pos_prob_4, neg_prob_4

    def norm_attention_predict(self, target_embedding, input_embedding):
        Q = input_embedding
        predict_1 = torch.matmul(input_embedding, target_embedding.transpose(0, 1))
        weight = torch.matmul(Q, target_embedding.transpose(0, 1))
        # scale
        weight = weight / (Q.shape[-1] ** 0.5)
        paddings = (torch.ones(weight.shape) * (-2 ** 32 + 1)).to(weight.device)
        weight_1 = torch.nn.Softmax(dim=1)(torch.where(torch.BoolTensor(weight.cpu() == 0.0).to(weight.device)
                                                       , paddings, weight))
        return ((predict_1 * weight_1).sum(1))

    def predict_period(self, uid, basket_1, basket_2, basket_3, basket_4, iid_1, iid_2, iid_3, iid_4,
                     neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4):

        # Aggregation Layer
        x = self.aggregate(basket_1, basket_2, basket_3, basket_4)

        # Attention Layer
        output_1, output_2, output_3, output_4, output_1_neg, output_2_neg, output_3_neg, output_4_neg = \
            self.encoder_norm_att(x, iid_1, iid_2, iid_3, iid_4,
                                  neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4)

        # Fusion Layer
        pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3, pos_prob_4, neg_prob_4 = self.fuse(uid,
                     output_1, output_2, output_3, output_4,output_1_neg, output_2_neg, output_3_neg, output_4_neg,
                     iid_1, iid_2, iid_3, iid_4, neg_iid_1, neg_iid_2, neg_iid_3, neg_iid_4)

        return pos_prob_1, neg_prob_1, pos_prob_2, neg_prob_2, pos_prob_3, neg_prob_3, pos_prob_4, neg_prob_4

    # predict_batch
    def compute_transpro_batch(self, uid, basket_1, basket_2, basket_3, basket_4):
        with torch.no_grad():
            basket_1 = pad_sequence([torch.tensor(i) for i in basket_1],
                                    batch_first=True, padding_value=self.n_items)
            basket_2 = pad_sequence([torch.tensor(i) for i in basket_2],
                                    batch_first=True, padding_value=self.n_items)
            basket_3 = pad_sequence([torch.tensor(i) for i in basket_3],
                                    batch_first=True, padding_value=self.n_items)
            basket_4 = pad_sequence([torch.tensor(i) for i in basket_4],
                                    batch_first=True, padding_value=self.n_items)

            basket_1 = basket_1.to(device=self.device)
            basket_2 = basket_2.to(device=self.device)
            basket_3 = basket_3.to(device=self.device)
            basket_4 = basket_4.to(device=self.device)

            x = self.aggregate(basket_1, basket_2, basket_3, basket_4)

            # exclude paddings
            item_emb_mor = self.IL_1.weight.data[:-1, :]
            item_emb_aft = self.IL_2.weight.data[:-1, :]
            item_emb_deep = self.IL_3.weight.data[:-1, :]
            item_emb_eve = self.IL_4.weight.data[:-1, :]

            mc_mor = self.norm_attention_predict(item_emb_mor, x)
            mc_aft = self.norm_attention_predict(item_emb_aft, x)
            mc_eve = self.norm_attention_predict(item_emb_eve, x)
            mc_deep = self.norm_attention_predict(item_emb_deep, x)

            # long-term
            x_mf_1 = torch.mm(self.UI_1.weight, self.IU.weight.t())[:, :-1].to(self.device)[uid]
            x_mf_2 = torch.mm(self.UI_2.weight, self.IU.weight.t())[:, :-1].to(self.device)[uid]
            x_mf_3 = torch.mm(self.UI_3.weight, self.IU.weight.t())[:, :-1].to(self.device)[uid]
            x_mf_4 = torch.mm(self.UI_4.weight, self.IU.weight.t())[:, :-1].to(self.device)[uid]

            # long-term and short-term fusion
            predict_mor = torch.sigmoid(self.alpha_mor) * mc_mor + (1 - torch.sigmoid(self.alpha_mor)) * x_mf_1
            predict_aft = torch.sigmoid(self.alpha_aft) * mc_aft + (1 - torch.sigmoid(self.alpha_aft)) * x_mf_2
            predict_eve = torch.sigmoid(self.alpha_eve) * mc_eve + (1 - torch.sigmoid(self.alpha_eve)) * x_mf_3
            predict_deep = torch.sigmoid(self.alpha_deep) * mc_deep + (1 - torch.sigmoid(self.alpha_deep)) * x_mf_4

            predict_mor_indices = torch.argsort(-predict_mor)
            predict_aft_indices = torch.argsort(-predict_aft)
            predict_eve_indices = torch.argsort(-predict_eve)
            predict_deep_indices = torch.argsort(-predict_deep)

            #Top20 itemset for each time segment respectively
            return predict_mor_indices[:, :20], predict_aft_indices[:, :20], \
                   predict_eve_indices[:, :20], predict_deep_indices[:, :20]
