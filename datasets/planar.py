import os.path as osp

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
import numpy as np
from scipy.spatial import Delaunay

from .utils import node_counts, save_pickle, load_pickle


class PlanarDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 8704
        self.n_nodes = 512
        super().__init__(root, transform, pre_transform, pre_filter)
        
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        data_list = []
        for i in range(self.n_graphs):
            # Generate planar graphs using Delauney traingulation
            points = np.random.rand(self.n_nodes, 2)
            tri = Delaunay(points)
            adj = np.zeros([self.n_nodes, self.n_nodes])
            for t in tri.simplices:
                adj[t[0], t[1]] = 1
                adj[t[1], t[2]] = 1
                adj[t[2], t[0]] = 1
                adj[t[1], t[0]] = 1
                adj[t[2], t[1]] = 1
                adj[t[0], t[2]] = 1

            adj = torch.from_numpy(adj).float()
            n = adj.shape[-1]
            x = torch.zeros(n, 1, dtype=torch.long)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        train_data_list = data_list[:8192]
        val_data_list = data_list[8192:8448]
        test_data_list = data_list[8448:]
        counts = {'train': 0, 'val': 0, 'test': 0}
        for split, data_list in zip(['train', 'val', 'test'], [train_data_list, val_data_list, test_data_list]):
            node_count = node_counts(data_list)
            counts[split] = node_count

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))
