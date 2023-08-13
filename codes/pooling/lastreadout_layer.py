import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter


class LastReadoutLayer(torch.nn.Module):
    def __init__(self, final_readout, pool_num_layer, emb_dim):
        super(LastReadoutLayer, self).__init__()
        self.final_readout = final_readout
        self.pool_num_layer = pool_num_layer
        self.emb_dim = emb_dim

        if self.final_readout in ["add", "mean", "max"]:
            self.readout_layer = getattr(
                globals()['global_' + self.final_readout + '_pool'], '__call__')
        elif self.final_readout in ["concate_add_pool", "concate_mean_pool", "concate_max_pool"]:
            self.projection_layer = torch.nn.Linear(
                (self.pool_num_layer+1)*self.emb_dim, self.emb_dim)
        # else:
        #     raise ValueError("Invalid last readout type.")

    def forward(self, hs, batches):
        if self.final_readout.startswith("concate"):
            g = getattr(
                globals()['global_' + self.final_readout.split('_')[1] + '_pool'], '__call__')(hs[0], batches[0])
            batch_size = g.shape[0]
            scatter_readout = self.final_readout.split(
                '_')[1] if self.final_readout.split('_')[1] != "add" else "sum"
            for layer in range(1, len(hs)):
                g = torch.cat([
                    g,
                    scatter(hs[layer], batches[layer], dim=0,
                            dim_size=batch_size, reduce=scatter_readout)
                ], dim=1)
            g = self.projection_layer(g)
        elif self.final_readout.startswith("combine"):
            g = getattr(
                globals()['global_' + self.final_readout.split('_')[1] + '_pool'], '__call__')(hs[0], batches[0])
            batch_size = g.shape[0]
            scatter_readout = self.final_readout.split(
                '_')[1] if self.final_readout.split('_')[1] != "add" else "sum"
            for layer in range(1, len(hs)):
                g += scatter(
                    hs[layer], batches[layer], dim=0, dim_size=batch_size, reduce=scatter_readout)
        else:
            g = self.readout_layer(hs[-1], batches[-1])
        return g
