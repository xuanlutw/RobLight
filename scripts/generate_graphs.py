from torch_geometric.datasets import AttributedGraphDataset, TUDataset, WebKB
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import os

graph_base = "./data/graph"

def do_node_dataset(dataset_name, r):
    if dataset_name in ["Cora", "CiteSeer"]:
        dataset = AttributedGraphDataset(root='/tmp/dataset', name=dataset_name)
        directed = "DIRECTED"
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='/tmp/dataset', name=dataset_name)
        directed = "DIRECTED"
    else:
        ValueError("Unknown dataset")
    data = dataset[0]
    G    = to_networkx(data)
    G    = G.reverse(copy=True)

    graph_path = os.path.join(graph_base, f"{dataset_name}")
    os.makedirs(graph_path, exist_ok=True)

    for i in range(G.number_of_nodes()):
        # slice
        Gi      = nx.ego_graph(G, i, radius=r)
        n_nodes = Gi.number_of_nodes()
        
        # rename
        labels = list(Gi.nodes)
        labels.remove(i)
        labels.insert(0, i)
        mapping = {j: idx for idx, j in enumerate(labels)}
        Gi = nx.relabel_nodes(Gi, mapping)

        # write graph
        with open(os.path.join(graph_path, f'{i}.graph'), 'w') as f:
            f.write(f"{n_nodes}\n")
            f.write(f"{directed}\n")
            for v in range(n_nodes):
                nbrs = list(Gi.neighbors(v))
                nbrs.sort()
                f.write(f"{len(nbrs)} {' '.join(str(x) for x in nbrs)}\n")
        
        # write feat
        feat = data.x[labels].numpy()
        np.savetxt(os.path.join(graph_path, f"{i}.feat"), feat, fmt='%d', delimiter=" ")

    print(f"{dataset_name} done!")

def do_graph_dataset(dataset_name):
    if dataset_name in ["MUTAG", "ENZYMES"]:
        data = TUDataset(root='../data/dataset', name=dataset_name)
        directed = "UNDIRECTED"
    else:
        ValueError("Unknown dataset")

    graph_path = os.path.join(graph_base, f"{dataset_name}")
    os.makedirs(graph_path, exist_ok=True)

    for i in range(len(data)):
        G = to_networkx(data[i])
        n_nodes = G.number_of_nodes()
        
        # write graph
        with open(os.path.join(graph_path, f'{i}.graph'), 'w') as f:
            f.write(f"{n_nodes}\n")
            f.write(f"{directed}\n")
            for v in range(n_nodes):
                nbrs = list(G.neighbors(v))
                nbrs.sort()
                f.write(f"{len(nbrs)} {' '.join(str(x) for x in nbrs)}\n")
        
        # write feat
        feat = data[i].x.numpy()
        np.savetxt(os.path.join(graph_path, f"{i}.feat"), feat, fmt='%d', delimiter=" ")

    print(f"{dataset_name} done!")

if __name__ == "__main__":
    do_node_dataset("Cora", 4)
    do_node_dataset("CiteSeer", 4)
    do_node_dataset("Cornell", 4)
    do_node_dataset("Texas", 4)
    do_node_dataset("Wisconsin", 4)

    do_graph_dataset("MUTAG")
    do_graph_dataset("ENZYMES")
