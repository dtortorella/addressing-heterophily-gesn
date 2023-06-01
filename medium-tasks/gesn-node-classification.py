import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import approximate_graph_spectral_radius, to_sparse_adjacency
from scipy.io import savemat
from torch.nn.functional import one_hot, linear
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor, Planetoid
from torch_geometric.utils import to_undirected
from tqdm import tqdm


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


def evaluate(metric, y_true, y_out):
    if metric == 'acc':
        return (y_true == y_out.argmax(dim=-1)).float().mean()
    elif metric == 'auc':
        y_true = y_true.detach().cpu().numpy()
        y_out = y_out.detach().cpu().numpy()
        return sum(roc_auc_score((y_true == i), y_out[:, i]) for i in range(y_out.shape[1])) / y_out.shape[1]


def validate_on(X_validation, y_validation, metric):
    return lambda weights: -evaluate(metric, y_validation, linear(X_validation, weights[0], weights[1]))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--undirected', help='whether to use the undirected graph', action='store_true')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[64])
parser.add_argument('--init', help='random recurrent initializer', type=str, choices=['uniform', 'normal', 'ring', 'orthogonal', 'symmetric', 'antisymmetric', 'diagonal'], default='uniform')
parser.add_argument('--iterations', help='max iterations', type=int, default=30)
parser.add_argument('--epsilon', help='convergence threshold', type=float, default=1e-8)
parser.add_argument('--rho', help='rho for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
parser.add_argument('--scale', help='scale for input matrix initialization', type=float, nargs='+', default=[1.0])
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
parser.add_argument('--trials', help='number of trials', type=int, default=5)
parser.add_argument('--no-bias', help='whether bias term is absent', action='store_false')
parser.add_argument('--no-feat', help='remove node features', action='store_true')
parser.add_argument('--batch', help='batch size for readout', type=int, default=None)
args = parser.parse_args()

dataset = get_dataset(args.root, args.dataset)
device = torch.device(args.device)
data = dataset[0].to(device)
y = one_hot(data.y, dataset.num_classes).float().to(device)
adj = to_sparse_adjacency(data.edge_index if not args.undirected else to_undirected(data.edge_index), num_nodes=data.num_nodes)

alpha = approximate_graph_spectral_radius(adj)
print(f'graph alpha = {float(alpha):.2f}')

train_acc = torch.zeros(data.train_mask.shape[1], len(args.units), len(args.rho), len(args.scale), args.trials)
val_acc = torch.zeros_like(train_acc)
test_acc = torch.zeros_like(train_acc)

if args.no_feat:
    data.x = torch.zeros(data.x.shape[0], 1).to(device)

with tqdm(total=train_acc.numel()) as progress:
    for unit_index, unit in enumerate(args.units):
        reservoir = StaticGraphReservoir(num_layers=1, in_features=data.x.shape[-1], hidden_features=unit,
                                         max_iterations=args.iterations, epsilon=args.epsilon, bias=args.no_bias)
        readout = Readout(num_features=reservoir.out_features, num_targets=dataset.num_classes)
        for rho_index, rho in enumerate(args.rho):
            for scale_index, scale in enumerate(args.scale):
                for trial_index in range(args.trials):
                    reservoir.initialize_parameters(recurrent=initializer(args.init, rho=(rho / alpha)),
                                                    input=initializer('uniform', scale=scale),
                                                    bias=initializer('uniform', scale=0.1))
                    reservoir.to(device)
                    x = reservoir(adj, data.x)
                    for mask_index in range(data.train_mask.shape[1]):
                        if args.batch is None:
                            fit_data = (x[data.train_mask[:, mask_index]], y[data.train_mask[:, mask_index]])
                        else:
                            fit_data = zip(torch.split(x[data.train_mask[:, mask_index]], args.batch),
                                           torch.split(y[data.train_mask[:, mask_index]], args.batch))
                        readout.fit(fit_data, args.ld, validate_on(x[data.val_mask[:, mask_index]], data.y[data.val_mask[:, mask_index]], 'acc'))
                        y_match = (readout(x).nan_to_num().argmax(dim=-1) == data.y)
                        train_acc[mask_index, unit_index, rho_index, scale_index, trial_index] = y_match[data.train_mask[:, mask_index]].float().mean()
                        val_acc[mask_index, unit_index, rho_index, scale_index, trial_index] = y_match[data.val_mask[:, mask_index]].float().mean()
                        test_acc[mask_index, unit_index, rho_index, scale_index, trial_index] = y_match[data.test_mask[:, mask_index]].float().mean()
                        progress.update(1)

savemat(f'{args.dataset}_{"undirected" if args.undirected else "directed"}_{args.init}{"_no-features" if args.no_feat else ""}.mat', mdict={
    'train_acc': train_acc.cpu().numpy(), 'val_acc': val_acc.cpu().numpy(), 'test_acc': test_acc.cpu().numpy(),
    'units': np.array(args.units), 'rho': np.array(args.rho), 'scale': np.array(args.scale), 'ld': np.array(args.ld)
})

