import torch
import torch.nn as nn

class OJFeatureEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_nodetypes, max_depth):
        super(OJFeatureEncoder, self).__init__()

        self.max_depth = max_depth
        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, node_type, depth):
        depth[depth > self.max_depth] = self.max_depth
        return (
            self.type_encoder(node_type)
            + self.depth_encoder(depth)
        )

class EdgeEncoder(torch.nn.Module):
    def __init__(self, dataset_name, in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        self.feature_layer = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.feature_layer(x)
