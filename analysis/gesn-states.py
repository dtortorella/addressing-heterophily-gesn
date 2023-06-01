import argparse

import numpy as np
import torch
from scipy.io import savemat
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor
from torch_geometric.utils import to_undirected, to_dense_adj

from graphesn import DynamicGraphReservoir, initializer
from graphesn.util import approximate_graph_spectral_radius, to_sparse_adjacency
from planetoid_dataset import Planetoid


class FakeSeq(list):
    def __init__(self, item, length):
        super().__init__()
        self.item = item
        self.length = length

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.item

    def __len__(self):
        return self.length


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
parser.add_argument('--directed', help='whether to use the directed graph', action='store_true')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--layers', help='reservoir layers', type=int, default=1)
parser.add_argument('--units', help='reservoir units per layer', type=int, default=64)
parser.add_argument('--init', help='random recurrent initializer', type=str,
                    choices=['uniform', 'normal', 'ring', 'orthogonal'], default='uniform')
parser.add_argument('--iterations', help='max iterations', type=int, default=100)
parser.add_argument('--rho', help='rho for recurrent matrix initialization', type=float, default=0.9)
parser.add_argument('--scale', help='scale for input matrix initialization', type=float, default=1.0)
parser.add_argument('--bias', help='whether bias term is present', action='store_true')
parser.add_argument('--chunks', help='split states into chunks', type=int, default=None)
parser.add_argument('--final', help='whether to get only the final state', action='store_true')
parser.add_argument('--fully', help='whether to get all layers in a deep model', action='store_true')
args = parser.parse_args()

dataset = get_dataset(args.root, args.dataset)
device = torch.device(args.device)
data = dataset[0].to(device)
adj = to_sparse_adjacency(data.edge_index if args.directed else to_undirected(data.edge_index),
                          num_nodes=data.num_nodes)
x_in = FakeSeq(data.x, args.iterations)

alpha = approximate_graph_spectral_radius(adj)
print(f'graph alpha = {float(alpha):.2f}')

reservoir = DynamicGraphReservoir(num_layers=args.layers, in_features=data.x.shape[-1], hidden_features=args.units,
                                  return_sequences=not args.final, fully=args.fully, bias=args.bias)
reservoir.initialize_parameters(recurrent=initializer(args.init, rho=(args.rho / alpha)),
                                input=initializer('uniform', scale=args.scale),
                                bias=initializer('uniform', scale=0.1))
reservoir.to(device)
X = reservoir(adj, x_in)


if args.chunks is None:
    if args.fully:
        X = np.array([x.cpu().numpy() for x in torch.split(X, args.units, dim=-1)])
    else:
        X = X.cpu().numpy()
    savemat(f'{args.dataset}_{"directed" if args.directed else "undirected"}_{args.init}_rho{args.rho:.1f}_K{args.iterations}.mat',
            mdict={'X': X})
else:
    X = torch.split(X, args.chunks)
    for index, x in enumerate(X):
        savemat(
            f'{args.dataset}_{"directed" if args.directed else "undirected"}_{args.init}_rho{args.rho:.1f}_K{args.iterations}_{index}.mat',
            mdict={'X': x.cpu().numpy()})
savemat(f'{args.dataset}_{"directed" if args.directed else "undirected"}_{args.init}_matrices.mat',
        mdict={'W': reservoir.layers[0].recurrent_weight.cpu().numpy(),
               'A': to_dense_adj(data.edge_index if args.directed else to_undirected(data.edge_index)).squeeze().cpu().numpy(),
               'Z': data.x.cpu().numpy()})
