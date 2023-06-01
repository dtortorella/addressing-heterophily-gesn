import argparse

import networkx as nx
from scipy.io import savemat
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor, Planetoid
from torch_geometric.utils import to_networkx


def get_dataset(root, name):
    if name in ['chameleon', 'squirrel']:
        return WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
    elif name in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=root, name=name)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=root, name=name, split='geom-gcn')
    elif name == 'actor':
        return Actor(root=root)
    else:
        raise ValueError(f'Unknown dataset `{name}`')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--undirected', help='whether to use the undirected graph', action='store_true')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
args = parser.parse_args()

dataset = get_dataset(args.root, args.dataset)
data = dataset[0]
G = to_networkx(data)
if args.undirected:
    G = G.to_undirected()

paths = nx.floyd_warshall_numpy(G)

savemat(f'{args.dataset}_paths_{"undirected" if args.undirected else "directed"}.mat', mdict={
    'P': paths
})
