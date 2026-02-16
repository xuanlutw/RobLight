from torch.nn import Module, ReLU, Linear
from torch_geometric.nn import SAGEConv, Sequential, global_add_pool

class NodeClassification(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, aggr, num_layers):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers >= 2")
        self.num_layers = num_layers

        # first layer: in -> hidden
        setattr(self, "layer1", Sequential(
            'x, edge_index',
            [
                (SAGEConv(in_channels, hidden_channels, aggr=aggr), 'x, edge_index -> x'),
                ReLU(inplace=True),
            ]
        ))

        # middle layer: hidden layers: hidden -> hidden
        for i in range(2, num_layers):
            setattr(self, f"layer{i}", Sequential(
                'x, edge_index',
                [
                    (SAGEConv(hidden_channels, hidden_channels, aggr=aggr), 'x, edge_index -> x'),
                    ReLU(inplace=True),
                ]
            ))

        # last layer: hidden -> out
        setattr(self, f"layer{num_layers}", Sequential(
            'x, edge_index',
            [
                (SAGEConv(hidden_channels, out_channels, aggr=aggr), 'x, edge_index -> x'),
            ]
        ))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"layer{i}")
            x     = layer(x, edge_index)

        return x

class GraphClassification(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, aggr, num_layers):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers >= 2")
        self.num_layers = num_layers

        # first layer: in -> hidden
        setattr(self, "layer1", Sequential(
            'x, edge_index',
            [
                (SAGEConv(in_channels, hidden_channels, aggr=aggr), 'x, edge_index -> x'),
                ReLU(inplace=True),
            ]
        ))

        # middle layer: hidden layers: hidden -> hidden
        for i in range(2, num_layers):
            setattr(self, f"layer{i}", Sequential(
                'x, edge_index',
                [
                    (SAGEConv(hidden_channels, hidden_channels, aggr=aggr), 'x, edge_index -> x'),
                    ReLU(inplace=True),
                ]
            ))

        # last layer: hidden -> hidden, without relu
        setattr(self, f"layer{num_layers}", Sequential(
            'x, edge_index',
            [
                (SAGEConv(hidden_channels, hidden_channels, aggr=aggr), 'x, edge_index -> x'),
            ]
        ))

        # linear layer: hidden -> out
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f"layer{i}")
            x     = layer(x, edge_index)

        x = global_add_pool(x, batch)
        x = self.lin(x)

        return x
