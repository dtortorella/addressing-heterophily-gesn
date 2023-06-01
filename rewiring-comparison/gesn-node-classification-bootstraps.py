import argparse
import math
import random

import numpy as np
import torch
from graphesn.util import approximate_graph_spectral_radius, to_sparse_adjacency
from scipy.io import savemat
from torch.nn.functional import one_hot
from tqdm import tqdm

from graphesn import StaticGraphReservoir, initializer, Readout

from gdl.data.base import BaseDataset
from gdl.seeds import test_seeds, development_seed


def set_train_val_test_split_frac(data, val_frac=.2, test_frac=.2):
    num_nodes = data.y.shape[0]

    data.train_mask = torch.zeros(num_nodes, len(test_seeds), dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, len(test_seeds), dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, len(test_seeds), dtype=torch.bool)

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    val_size = math.ceil(val_frac * num_nodes)
    test_size = math.ceil(test_frac * num_nodes)
    train_size = num_nodes - val_size - test_size

    for seed_index, seed in enumerate(test_seeds):
        nodes = list(range(num_nodes))

        # Take same test set every time using development seed for robustness
        random.seed(development_seed)
        random.shuffle(nodes)
        test_idx = sorted(nodes[:test_size])
        nodes = [x for x in nodes if x not in test_idx]

        # Take train / val split according to seed
        random.seed(seed)
        random.shuffle(nodes)
        train_idx = sorted(nodes[:train_size])
        val_idx = sorted(nodes[train_size:])

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes

        data.train_mask[:, seed_index] = get_mask(train_idx)
        data.val_mask[:, seed_index] = get_mask(val_idx)
        data.test_mask[:, seed_index] = get_mask(test_idx)

    return data


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--undirected', help='whether to use the undirected graph', action='store_true')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[64])
parser.add_argument('--init', help='random recurrent initializer', type=str, choices=['uniform', 'normal', 'ring', 'orthogonal'], default='uniform')
parser.add_argument('--iterations', help='max iterations', type=int, default=100)
parser.add_argument('--epsilon', help='convergence threshold', type=float, default=1e-8)
parser.add_argument('--rho', help='rho for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
parser.add_argument('--scale', help='scale for input matrix initialization', type=float, nargs='+', default=[1.0])
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
parser.add_argument('--trials', help='number of trials', type=int, default=5)
parser.add_argument('--bias', help='whether bias term is present', action='store_true')
parser.add_argument('--no-feat', help='remove node features', action='store_true')
args = parser.parse_args()

dataset = BaseDataset(name=args.dataset, data_dir=args.root, undirected=args.undirected)
device = torch.device(args.device)
data = set_train_val_test_split_frac(dataset[0]).to(device)
y = one_hot(data.y, dataset.num_classes).float().to(device)
adj = to_sparse_adjacency(data.edge_index, num_nodes=data.num_nodes).t()

alpha = approximate_graph_spectral_radius(adj)
print(f'graph alpha = {float(alpha):.2f}')

train_acc = torch.zeros(data.train_mask.shape[1], len(args.units), len(args.rho), len(args.scale), len(args.ld), args.trials)
val_acc = torch.zeros_like(train_acc)
test_acc = torch.zeros_like(train_acc)

if args.no_feat:
    data.x = torch.zeros(data.x.shape[0], 1).to(device)

with tqdm(total=train_acc.numel()) as progress:
    for unit_index, unit in enumerate(args.units):
        reservoir = StaticGraphReservoir(num_layers=1, in_features=data.x.shape[-1], hidden_features=unit,
                                         max_iterations=args.iterations, epsilon=args.epsilon, bias=args.bias)
        readout = Readout(num_features=reservoir.out_features, num_targets=dataset.num_classes)
        for rho_index, rho in enumerate(args.rho):
            for scale_index, scale in enumerate(args.scale):
                for trial_index in range(args.trials):
                    reservoir.initialize_parameters(recurrent=initializer(args.init, rho=(rho / alpha)),
                                                    input=initializer('uniform', scale=scale),
                                                    bias=initializer('uniform', scale=0.1))
                    reservoir.to(device)
                    x = reservoir(adj, data.x)
                    for ld_index, ld in enumerate(args.ld):
                        for mask_index in range(data.train_mask.shape[1]):
                            try:
                                readout.fit((x[data.train_mask[:, mask_index]], y[data.train_mask[:, mask_index]]), ld)
                                y_match = (readout(x).argmax(dim=-1) == data.y)
                                train_acc[mask_index, unit_index, rho_index, scale_index, ld_index, trial_index] = y_match[data.train_mask[:, mask_index]].float().mean()
                                val_acc[mask_index, unit_index, rho_index, scale_index, ld_index, trial_index] = y_match[data.val_mask[:, mask_index]].float().mean()
                                test_acc[mask_index, unit_index, rho_index, scale_index, ld_index, trial_index] = y_match[data.test_mask[:, mask_index]].float().mean()
                            except:
                                pass
                            progress.update(1)

savemat(f'{args.dataset}_{"directed" if not args.undirected else "undirected"}_{args.init}{"_no-features" if args.no_feat else ""}.mat', mdict={
    'train_acc': train_acc.cpu().numpy(), 'val_acc': val_acc.cpu().numpy(), 'test_acc': test_acc.cpu().numpy(),
    'units': np.array(args.units), 'rho': np.array(args.rho), 'scale': np.array(args.scale), 'ld': np.array(args.ld)
})

