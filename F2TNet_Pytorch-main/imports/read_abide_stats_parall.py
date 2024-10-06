'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import os
import glob
import h5py

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
# from networkx.convert_matrix import from_numpy_matrix
from networkx.convert_matrix import from_numpy_array
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
from imports.gdc import GDC


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices

def read_datat1w(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_hcpt1w_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices

def read_datat1w_score(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_hcpt1w_data_score, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices

def read_datafmri_score(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_hcpfmri_data_score, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    y_arr = (y_arr - np.mean(y_arr)) / np.std(y_arr)
    # y_arr = (y_arr - np.min(y_arr)) / (np.max(y_arr) - np.min(y_arr))

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr)  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices


def read_datafmrit1w_score(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    y2_list = []
    y3_list = []
    y4_list = []
    y5_list = []
    y6_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_hcpfmrit1w_data_score, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    sbj_label_empty = np.zeros((len(res),1))
    for j in range(len(res)):
        if res[j][3].size==0:
            sbj_label_empty[j,0] = 1
        if res[j][5].size==0:
            sbj_label_empty[j,0] = 1
        if res[j][6].size==0:
            sbj_label_empty[j,0] = 1
        if res[j][7].size==0:
            sbj_label_empty[j,0] = 1
        if res[j][8].size==0:
            sbj_label_empty[j,0] = 1
        if res[j][9].size==0:
            sbj_label_empty[j,0] = 1



    delete_index = np.where(sbj_label_empty==1)
    if delete_index[0].size !=0:
        delete_index = delete_index[0].reshape((delete_index[0].size,1))
        delete_index  = delete_index[::-1]
        for j in range(delete_index.size):
            del res[int(delete_index[j])]


    # delete_index = delete_index[0].reshape((5,1))
    # delete_index  = delete_index[::-1]
    # for j in range(delete_index.size):
    #     del res[int(delete_index[j])]





    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))
        y2_list.append(res[j][5])
        y3_list.append(res[j][6])
        y4_list.append(res[j][7])
        y5_list.append(res[j][8])
        y6_list.append(res[j][9])

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    y_arr = (y_arr - np.mean(y_arr)) / np.std(y_arr)
    y2_arr = np.stack(y2_list)
    y2_arr = (y2_arr - np.mean(y2_arr)) / np.std(y2_arr)
    y3_arr = np.stack(y3_list)
    y3_arr = (y3_arr - np.mean(y3_arr)) / np.std(y3_arr)
    y4_arr = np.stack(y4_list)
    y4_arr = (y4_arr - np.mean(y4_arr)) / np.std(y4_arr)
    y5_arr = np.stack(y5_list)
    y5_arr = (y5_arr - np.mean(y5_arr)) / np.std(y5_arr)
    y6_arr = np.stack(y6_list)
    y6_arr = (y6_arr - np.mean(y6_arr)) / np.std(y6_arr)

    y_arr = np.concatenate((y_arr, y2_arr, y3_arr, y4_arr, y5_arr, y6_arr), axis=1)
    # y_arr = (y_arr - np.min(y_arr)) / (np.max(y_arr) - np.min(y_arr))

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr)  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices


def read_datafmri_score4class(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_hcpfmri_data_score_class4, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    y_arr = (y_arr - np.mean(y_arr)) / np.std(y_arr)
    # y_arr = (y_arr - np.min(y_arr)) / (np.max(y_arr) - np.min(y_arr))

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr)  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices


def read_sigle_data(data_dir,filename,use_gdc =False):

    temp = dd.io.load(osp.join(data_dir, filename))

    # read edge and edge attribute
    pcorr = np.abs(temp['pcorr'][()])

    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    label = temp['label'][()]

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes


def read_sigle_hcpt1w_data(data_dir,filename,use_gdc =False):

    temp = dd.io.load(osp.join(data_dir, filename))
    feature = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_feature_Gordon333.txt'))
    adj_matrix = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_adj_Gordon333.txt'))

    # read edge and edge attribute


    pcorr = np.abs(adj_matrix)
    np.fill_diagonal(pcorr, np.inf)




    num_nodes = pcorr.shape[0]


    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = feature #temp['corr'][()]
    label = temp['label'][()]

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes


def read_sigle_hcpt1w_data_score(data_dir,filename,use_gdc =False):

    temp = dd.io.load(osp.join(data_dir, filename))
    feature = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_feature_Gordon333.txt'))
    adj_matrix = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_adj_Gordon333.txt'))
    label_all = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    # read edge and edge attribute


    pcorr = np.abs(adj_matrix)
    np.fill_diagonal(pcorr, np.inf)

    num_nodes = pcorr.shape[0]


    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = feature #temp['corr'][()]
    # label = temp['label'][()]
    label = label_all[np.where(label_all==int(filename[:-3]))[0],1]

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes


def read_sigle_hcpfmri_data_score(data_dir,filename,use_gdc =False):
    temp = dd.io.load(osp.join(data_dir, filename))
    label_all = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    # ProcSpeed_AgeAdj.txt     ReadEng_Unadj.txt PMAT24_A_CR.txt
    # read edge and edge attribute
    pcorr = temp['pcorr'][()]
    pcorr[pcorr < 0] = 0
    positive_values = pcorr[pcorr > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    pcorr[pcorr < threshold] = 0

    # flattened_matrix = pcorr.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(pcorr, top_values)
    # pcorr_1 = np.where(mask, pcorr, 0)
    # pcorr = pcorr_1



    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    att[att < 0] = 0
    positive_values = att[att > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    att[att < threshold] = 0


    # num_elements = int(att.size * 0.15)
    # flattened_matrix = att.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(att, top_values)
    # att_1 = np.where(mask, att, 0)
    # att = att_1


    label = label_all[np.where(label_all==int(filename[:-3]))[0],1]



    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes



def read_sigle_hcpfmrit1w_data_score(data_dir,filename,use_gdc =False):


    temp = dd.io.load(osp.join(data_dir, filename))
    feature = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_feature_Gordon333.txt'))
    adj_matrix = np.genfromtxt(osp.join(data_dir[:-73],'/data/hzb/project/Brain_Predict_Score/data_HCP_0114/T1w_9/T1w_data_preprocess_0114', filename[:-3], filename[:-3]+'_ROI_adj_Gordon333.txt'))
    label_all = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    # read edge and edge attribute


    pcorr = np.abs(adj_matrix)
    np.fill_diagonal(pcorr, np.inf)

    num_nodes = pcorr.shape[0]


    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = feature #temp['corr'][()]
    # label = temp['label'][()]
    label = label_all[np.where(label_all==int(filename[:-3]))[0],1]

    att_torch = torch.from_numpy(att).float()

    t1w_feature = att









    temp = dd.io.load(osp.join(data_dir, filename))
    label_all = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    # label_1 = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    # label_2 = np.genfromtxt(osp.join(data_dir[:-25], 'PicVocab_Unadj.txt'))
    # label_3 = np.genfromtxt(osp.join(data_dir[:-25], 'VSPLOT_TC.txt'))




    label_1 = np.genfromtxt(osp.join(data_dir[:-25], 'VSPLOT_TC.txt'))
    label_2 = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))
    label_3 = np.genfromtxt(osp.join(data_dir[:-25], 'PercStress_Unadj.txt'))
    label_4 = np.genfromtxt(osp.join(data_dir[:-25], 'AngAggr_Unadj.txt'))
    label_5 = np.genfromtxt(osp.join(data_dir[:-25], 'Strength_Unadj.txt'))
    label_6 = np.genfromtxt(osp.join(data_dir[:-25], 'Endurance_Unadj.txt'))


    # ProcSpeed_AgeAdj.txt     ReadEng_Unadj.txt PMAT24_A_CR.txt
    # read edge and edge attribute
    pcorr = temp['pcorr'][()]
    pcorr[pcorr < 0] = 0
    positive_values = pcorr[pcorr > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    pcorr[pcorr < threshold] = 0

    # flattened_matrix = pcorr.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(pcorr, top_values)
    # pcorr_1 = np.where(mask, pcorr, 0)
    # pcorr = pcorr_1



    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    att[att < 0] = 0
    positive_values = att[att > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    att[att < threshold] = 0

    att = np.concatenate((t1w_feature, att), axis=1)
    # num_elements = int(att.size * 0.15)
    # flattened_matrix = att.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(att, top_values)
    # att_1 = np.where(mask, att, 0)
    # att = att_1


    label = label_all[np.where(label_all==int(filename[:-3]))[0],1]

    label_sub1 =label_1[np.where(label_1==int(filename[:-3]))[0],1]
    label_sub2 =label_2[np.where(label_2==int(filename[:-3]))[0],1]
    label_sub3 =label_3[np.where(label_3==int(filename[:-3]))[0],1]

    label_sub4 =label_4[np.where(label_4==int(filename[:-3]))[0],1]
    label_sub5 =label_5[np.where(label_5==int(filename[:-3]))[0],1]
    label_sub6 =label_6[np.where(label_6==int(filename[:-3]))[0],1]

    # label = np.array([label_sub1, label_sub2, label_sub3]).T
    # label = label.astype(float)




    att_torch = torch.from_numpy(att).float()
    # att_torch = torch.cat((t1w_feature, att_torch), dim=1)
    y_torch = torch.from_numpy(np.array(label)).long()  # classification
    # y_torch =    torch.from_numpy(np.array([label_sub1, label_sub2, label_sub3])).long()

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label_sub1,num_nodes, label_sub2, label_sub3, label_sub4, label_sub5, label_sub6



def read_sigle_hcpfmri_data_score_class4(data_dir,filename,use_gdc =False):
    temp = dd.io.load(osp.join(data_dir, filename))
    label_all = np.genfromtxt(osp.join(data_dir[:-25], 'ReadEng_Unadj.txt'))

    # read edge and edge attribute
    pcorr = temp['pcorr'][()]
    pcorr[pcorr < 0] = 0
    positive_values = pcorr[pcorr > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    pcorr[pcorr < threshold] = 0

    # flattened_matrix = pcorr.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(pcorr, top_values)
    # pcorr_1 = np.where(mask, pcorr, 0)
    # pcorr = pcorr_1



    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    att[att < 0] = 0
    positive_values = att[att > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    att[att < threshold] = 0


    # num_elements = int(att.size * 0.15)
    # flattened_matrix = att.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(att, top_values)
    # att_1 = np.where(mask, att, 0)
    # att = att_1


    label = label_all[np.where(label_all==int(filename[:-3]))[0],1]



    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes




if __name__ == "__main__":
    data_dir = '/home/azureuser/projects/BrainGNN/data/ABIDE_pcp/cpac/filt_noglobal/raw'
    filename = '50346.h5'
    read_sigle_data(data_dir, filename)






