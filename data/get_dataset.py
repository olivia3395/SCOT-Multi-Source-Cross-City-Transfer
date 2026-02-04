import os
from abc import ABC

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F


def load_data(dataset):
    data_path = './raw_data'
    region_data = pd.read_csv(data_path + '/{}/regionmap_{}/regionmap_{}.geo'.format(dataset, dataset, dataset))
    region_mob = pd.read_csv(data_path + '/{}/regionmap_{}/regionmap_{}.mob'.format(dataset, dataset, dataset))
    mob = np.load(data_path + '/{}/region_od_flow_{}_11.npy'.format(dataset, dataset))
    mob = mob / np.sum(mob)
    geo_to_ind, ind_to_geo = {}, {}
    geo_ids = list(region_data['geo_id'])
    for index, geo_id in enumerate(geo_ids):
        geo_to_ind[geo_id] = index
        ind_to_geo[index] = geo_id

    region_num = len(region_data['geo_id'])
    mob_file = region_mob[['origin_id', 'destination_id', 'mobility_weight']]
    adj_mx_mob = np.zeros((region_num, region_num), dtype=np.float32)

    for row in mob_file.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            continue
        adj_mx_mob[geo_to_ind[row[0]], geo_to_ind[row[1]]] = row[2]

    identity_matrix = np.eye(len(adj_mx_mob), dtype=np.float32)
    adj_mx_mob_sparse = torch.tensor(adj_mx_mob + identity_matrix).to_sparse_coo()
    # adj_mx_mob_sparse = torch.tensor(adj_mx_mob).to_sparse_coo()
    edge_index, edge_weight = adj_mx_mob_sparse.indices(), adj_mx_mob_sparse.values()

    graph_info = {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'mob': mob
    }
    return region_num, graph_info


