from torch_geometric.data import Dataset as geometricDataset
from torch_geometric.data import Data
import torch

import pickle
import gzip
import random
import numpy
from tqdm import tqdm
import copy


def extract_OJdataset(data_path, valid_ratio=0.2, vocab_size=5000, depth_augment=True, depth_group=3):
    train_set, test_set, valid_set = [], [], []
    with gzip.open(data_path, "rb") as f:
        dic = pickle.load(f)

    # get vocabulary
    idx2node = dic['idx2node'][:vocab_size]
    node2idx = {"<pad>": 0}
    for i, t in zip(range(vocab_size), idx2node):
        node2idx[t] = i
        assert node2idx[t] == dic['node2idx'][t]

    # get num of train/val/test
    idxs = random.sample(range(len(dic['node_tr'])), len(dic['node_tr']))
    n_valid = int(len(dic['node_tr']) * valid_ratio)
    n_train = len(dic['node_tr']) - n_valid
    n_test = len(dic["node_te"])
    print("train num: ", n_train, "valid num:", n_valid, "test_num:", n_test)
    # valid
    for i in tqdm(idxs[:n_valid], mininterval=30):
        for node in range(len(dic["node_tr"][i])):
            if dic["node_tr"][i][node] >= vocab_size:
                dic["node_tr"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_tr"][i]), dtype=torch.int)
        for node in range(len(dic["begin_tr"][i])):
            curnode = dic["begin_tr"][i][node]
            fathernode = dic["end_tr"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)

        valid_data = Data(
            node_type=torch.tensor(dic["node_tr"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_tr"][i], dic["end_tr"][i]])),
            edge_attr=torch.ones(len(dic["begin_tr"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_tr'][i]]),
            ids=torch.tensor([i])
        )
        valid_set.append(valid_data)
    # train
    # for i in tqdm(idxs[n_valid:2*n_valid], mininterval=30):
    for i in tqdm(idxs[n_valid:], mininterval=30):
        for node in range(len(dic["node_tr"][i])):
            if dic["node_tr"][i][node] >= vocab_size:
                dic["node_tr"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_tr"][i]), dtype=torch.int)
        for node in range(len(dic["begin_tr"][i])):
            curnode = dic["begin_tr"][i][node]
            fathernode = dic["end_tr"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)
        train_data = Data(
            node_type=torch.tensor(dic["node_tr"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_tr"][i], dic["end_tr"][i]])),
            edge_attr=torch.ones(len(dic["begin_tr"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_tr'][i]]),
            ids=torch.tensor([i])
        )
        train_set.append(train_data)
    # test
    for i in tqdm(range(n_test), mininterval=30):
        for node in range(len(dic["node_te"][i])):
            if dic["node_te"][i][node] >= vocab_size:
                dic["node_te"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_te"][i]), dtype=torch.int)
        for node in range(len(dic["begin_te"][i])):
            curnode = dic["begin_te"][i][node]
            fathernode = dic["end_te"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)
        test_data = Data(
            node_type=torch.tensor(dic["node_te"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_te"][i], dic["end_te"][i]])),
            edge_attr=torch.ones(len(dic["begin_te"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_te'][i]]),
            ids=torch.tensor([i])
        )
        test_set.append(test_data)
    return train_set, valid_set, test_set, len(idx2node)


def get_load_dataset(data_list):
    def load_graph_dataset():
        return OJDataset(data_list)
    return load_graph_dataset


class OJDatasetLoader():
    def load_dataset(data_list):
        dataset = OJDataset(data_list)
        return dataset


class OJDataset(geometricDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.data_len


def extract_small_OJdataset(data_path, valid_ratio=0.2, vocab_size=5000, depth_augment=True, depth_group=3):
    # for debug or quick try usage
    train_set, test_set, valid_set = [], [], []
    with gzip.open(data_path, "rb") as f:
        dic = pickle.load(f)

    # get vocabulary
    idx2node = dic['idx2node'][:vocab_size]
    node2idx = {"<pad>": 0}
    for i, t in zip(range(vocab_size), idx2node):
        node2idx[t] = i
        assert node2idx[t] == dic['node2idx'][t]

    # get num of train/val/test
    idxs = random.sample(range(len(dic['node_tr'])), len(dic['node_tr']))
    n_valid = int(int(len(dic['node_tr']) * valid_ratio) * 0.1)
    n_train = int((len(dic['node_tr']) - n_valid)*0.1)
    n_test = int(len(dic["node_te"])*0.1)
    print("train num: ", n_train, "valid num:", n_valid, "test_num:", n_test)
    # valid
    for i in tqdm(idxs[:n_valid], mininterval=30):
        for node in range(len(dic["node_tr"][i])):
            if dic["node_tr"][i][node] >= vocab_size:
                dic["node_tr"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_tr"][i]), dtype=torch.int)
        for node in range(len(dic["begin_tr"][i])):
            curnode = dic["begin_tr"][i][node]
            fathernode = dic["end_tr"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)

        valid_data = Data(
            node_type=torch.tensor(dic["node_tr"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_tr"][i], dic["end_tr"][i]])),
            edge_attr=torch.ones(len(dic["begin_tr"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_tr'][i]]),
            ids=torch.tensor([i])
        )
        valid_set.append(valid_data)
    # train
    for i in tqdm(idxs[n_valid:n_valid+n_train], mininterval=30):
        for node in range(len(dic["node_tr"][i])):
            if dic["node_tr"][i][node] >= vocab_size:
                dic["node_tr"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_tr"][i]), dtype=torch.int)
        for node in range(len(dic["begin_tr"][i])):
            curnode = dic["begin_tr"][i][node]
            fathernode = dic["end_tr"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)
        train_data = Data(
            node_type=torch.tensor(dic["node_tr"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_tr"][i], dic["end_tr"][i]])),
            edge_attr=torch.ones(len(dic["begin_tr"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_tr'][i]]),
            ids=torch.tensor([i])
        )
        train_set.append(train_data)
    # test
    for i in tqdm(range(n_test), mininterval=30):
        for node in range(len(dic["node_te"][i])):
            if dic["node_te"][i][node] >= vocab_size:
                dic["node_te"][i][node] = node2idx["<unk>"]
        node_depth = torch.ones(len(dic["node_te"][i]), dtype=torch.int)
        for node in range(len(dic["begin_te"][i])):
            curnode = dic["begin_te"][i][node]
            fathernode = dic["end_te"][i][node]
            node_depth[curnode] = node_depth[fathernode] + 1
        origin_node_depth = copy.deepcopy(node_depth)
        if depth_augment == True:
            max_depth = int(max(node_depth))
            if(max_depth > depth_group):
                interval = int(max_depth / depth_group)
                # update node depth
                for d in range(0, depth_group):
                    left = d * interval
                    right = (d+1) * interval
                    if(d == depth_group-1):
                        right = max_depth+1
                    node_idxs = torch.where(
                        (node_depth >= left) & (node_depth < right))[0]
                    node_depth[node_idxs] = torch.tensor(
                        [d] * node_idxs.shape[0], dtype=torch.int)
        test_data = Data(
            node_type=torch.tensor(dic["node_te"][i]),
            edge_index=torch.tensor(numpy.stack(
                [dic["begin_te"][i], dic["end_te"][i]])),
            edge_attr=torch.ones(len(dic["begin_te"][i]), 1),
            node_depth=node_depth,
            origin_node_depth=origin_node_depth,
            y=torch.tensor([dic['y_te'][i]]),
            ids=torch.tensor([i])
        )
        test_set.append(test_data)
    return train_set, valid_set, test_set, len(idx2node)


if __name__ == '__main__':
    path = "data/OJ104/oj_ast_graph.pkl.gz"
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
        # dict_keys(['raw_tr', 'y_tr', 'node_tr', 'begin_tr',
        #           'end_tr', 'raw_te', 'y_te', 'node_te', 'begin_te',
        #           'end_te', 'node2idx', 'idx2node'])
