from torch_geometric.nn.pool.topk_pool import filter_adj
import torch
from layers.mlp_layer import MLP


class PoolingLayer(torch.nn.Module):
    def __init__(self, params, **kwargs):
        super(PoolingLayer, self).__init__()
        self.graph_pooling = params["pooling"]
        self.embed_dim = params["emb_dim"]
        if self.graph_pooling == "treepool":
            self.pool = TreePool(params["treepool"], self.embed_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, graph_indicator, edge_index, edge_attr, node_depth):
        return self.pool(x, node_depth, graph_indicator, edge_index, edge_attr)

    def pool_loss(self):
        return self.pool.get_pool_layer_loss()


class TreePool(torch.nn.Module):
    def __init__(self, params, embed_dim):
        super(TreePool, self).__init__()
        print("className", self.__class__.__name__)
        self.input_dim = embed_dim
        ggd_type = params["ggd_type"]
        ggd_readout = params["ggd_readout"]
        ggd_hidden_dim = params["ggd_hidden_dim"]
        if params["activation"] == "tanh":
            self.activation = torch.tanh

        if ggd_type == "GGD":
            self.mutual_layer = GGD(
                self.input_dim, ggd_hidden_dim, "prelu", ggd_readout)

    def forward(self, input_feature, node_depth, graph_indicator, edge_index, edge_attr=None):
        # 1. find the mask node(keeping as next pool)
        max_depth = max(node_depth)
        new_nodes = torch.where(node_depth < max_depth)[
            0].to(edge_index.device)
        # 2. get new feature
        pos = input_feature[new_nodes]
        neg = torch.zeros_like(pos)
        # neg with feature change
        neg = pos[:, torch.randperm(pos.shape[1]).to(pos.device)]
        self.ret = self.mutual_layer(pos, neg, input_feature)

        # 3. get new graph
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, new_nodes, num_nodes=graph_indicator.size(0)
        )
        return (
            input_feature[new_nodes],
            edge_index,
            edge_attr,
            graph_indicator[new_nodes],
            node_depth[new_nodes],
        )

    def get_pool_layer_loss(self):
        num = int(self.ret.shape[1] / 2)
        lbl = torch.cat(
            (torch.ones(1, num), torch.zeros(1, num)), 1
        ).to(self.ret.device)
        b_xent = torch.nn.BCEWithLogitsLoss()
        mutual_loss = b_xent(self.ret, lbl)
        return mutual_loss


class GGD(torch.nn.Module):
    def __init__(self, n_in, n_h, activation, readout="mean"):
        super(GGD, self).__init__()
        self.mlp = MLP(n_in, n_h, activation)
        self.global_mlp = MLP(n_in, n_h, activation)
        self.readout = readout

    def forward(self, seq1, seq2, node_emb):
        h_pos = self.mlp(seq1)
        h_neg = self.mlp(seq2)
        if self.readout == "mean":
            g_emb = torch.mean(node_emb, dim=0, keepdim=True)
        elif self.readout == "max":
            g_emb, _ = torch.max(node_emb, dim=0, keepdim=True)
        elif self.readout == "sum":
            g_emb = torch.sum(node_emb, dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Invalid readout option: {self.readout}. Try 'mean', 'max' or 'sum'")
        g_emb = self.global_mlp(g_emb)
        sc_1 = (h_pos+g_emb).sum(1).unsqueeze(0)
        sc_2 = (h_neg+g_emb).sum(1).unsqueeze(0)
        logits = torch.cat((sc_1, sc_2), 1)
        return logits
