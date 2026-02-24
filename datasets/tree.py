import os.path as osp

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
import networkx as nx


from .utils import node_counts, save_pickle, load_pickle


class TreeDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 8704
        self.n_nodes = 64
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
            G = nx.random_tree(self.n_nodes)
            adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
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
