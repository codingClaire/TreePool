import torch
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from pooling.pooling_layer import PoolingLayer
from pooling.lastreadout_layer import LastReadoutLayer
from layers.encoders import OJFeatureEncoder


class Model(torch.nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.pool_num_layer = params["pool_num_layer"]
        self.graph_pooling = params["pooling"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        self.final_readout = params["final_readout"]

        # check validation
        if self.gnn_type not in ["gcn", "gin", "gat", "graphsage"]:
            raise ValueError("Invalid GNN type.")
        self.feature_encoder = OJFeatureEncoder(
            self.emb_dim,
            params["num_nodetypes"],
            params["max_depth"],
        )
        self.gnnLayers = torch.nn.ModuleList()
        self.poolLayers = torch.nn.ModuleList()
        for _ in range(self.pool_num_layer):
            ### 1.GNN to generate node embeddings ###
            if self.virtual_node == "True":
                self.gnnLayers.append(GnnLayerwithVirtualNode(params))
            else:
                self.gnnLayers.append(GnnLayer(params))
            ### 2.Pooling method to generate pooling of graph ###
            self.poolLayers.append(PoolingLayer(params))

        self.readoutLayer = LastReadoutLayer(
            self.final_readout, self.pool_num_layer, self.emb_dim)

        ### 3.Prediction ###
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(
                2 * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        # 1. encode input feature
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        batch, node_depth = batched_data.batch, batched_data.node_depth
        input_feature = self.feature_encoder(
            batched_data.node_type, batched_data.origin_node_depth)
        # 2. (gnn layer * num_layer times + pool layer) * pool_num_layer times

        h = input_feature
        # hs = [h]
        # batches = [batch]
        hs = []
        batches = []

        for i in range(self.pool_num_layer):
            if self.virtual_node == "True":
                h = self.gnnLayers[i](h, edge_index, edge_attr, batch)
            else:
                h = self.gnnLayers[i](h, edge_index, edge_attr)
            hs.append(h)
            batches.append(batch)
            h, edge_index, edge_attr, batch, node_depth = self.poolLayers[i](
                h, batch, edge_index, edge_attr, node_depth)
        hs.append(h)
        batches.append(batch)

        model_loss = self.get_model_loss()
        # 3.final readout layer
        graph_representation = self.readoutLayer(hs, batches)
        # 4. prediction
        pred = self.graph_pred_linear(graph_representation)
        return pred, model_loss

    def get_model_loss(self):
        model_loss = 0
        for i in range(self.pool_num_layer):
            model_loss += self.poolLayers[i].pool_loss()
        return model_loss
