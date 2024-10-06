import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# from scipy import signal
# from scipy.fftpack import fft
# from pylab import *


def contrastive_node_module(t1, t2):
    contrast_loss = 0.0
    # i1 = 0, 0
    bs, node_num, _len = t1.shape
    for i in range(node_num):
        # print(i)
        # i1 += 1
        data_list = []
        pos_vec = t1[:,i,...] # [b,1024]
        data_list.append(pos_vec)
        neg_vec = t2[:,i,...] # [b,1024]
        data_list.append(neg_vec)


        neg_vec = torch.cat((t2[:, :i,:], t2[:, i+1:,:]), dim=1)
        neg_vec_list = list(tuple(neg_vec[:,j,:] for j in range(neg_vec.size(1))))
        data_list = data_list + neg_vec_list


        # j=1
        # neg_vec_sub = neg_vec[:, j, ...]
        # data_list.append(neg_vec_sub)
        #
        # # for j in range(neg_vec.shape[1]):
        # #     neg_vec_sub = neg_vec[:,j,...]
        # #     data_list.append(neg_vec_sub)
        contrast_loss += contrastive_loss(data_list)

    return contrast_loss /(node_num)






# def contrastive_module(self, data, t=0.07):  # T=0.07 data=[b,4,128,8]
#     contrast_loss = 0.0
#     i1, i2 = 0, 0
#     bs ,c ,piece_num ,_len = data.shape  # piece_num=128
#
#     for i in range(c):
#         pos_vec = data[: ,i ,...]  # [b,128,8]
#         for j in range(4 ,piece_num/ / 2 -1 ,16):
#             i1 += 1
#             data_list = []
#             data_list.append(pos_vec[: ,j ,:])
#             data_list.append(pos_vec[: , j +piece_num/ /4 ,:])
#             neg_list = [data[: ,k ,...][: , j -4: j +5 ,:] for k in range(c) if k != i]
#             data_list.extend(neg_list[idx][: ,m ,:] for idx in range( c -1) for m in range(neg_list[0].shape[1]))
#             contrast_loss += self.contrastive_loss(data_list, t)
#
#     return contrast_loss /(i1 +i2)


def contrastive_loss( data_list, t=0.07):  # data_list=[pos1,pos2,neg...]
    '''
    Compute softmax-based contrastive loss with temperature t (like Info-NCE)
    '''
    pos_score = score_t(data_list[0], data_list[1], t)
    all_score = 0.0

    for i in range(1 ,len(data_list)):
        all_score += score_t(data_list[0], data_list[i], t)
    contrast = - torch.log(pos_score /all_score +1e-5).mean()

    return contrast


def score_t(x, y, t=0.07):  # x=[b,8]
    '''
    Compute the similarity score between x and y with temperature t
    '''
    if torch.norm(x ,dim=1).mean().item() <=0.001 or torch.norm(y ,dim=1).mean().item() <=0.001:
        print (torch.norm(x ,dim=1).mean().item() ,torch.norm(y ,dim=1).mean().item())

    return torch.exp(( x *y).sum(1 ) /( t *(torch.norm(x ,dim=1 ) *torch.norm(y ,dim=1) ) +1e-5))