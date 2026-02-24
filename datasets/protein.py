import os
import pathlib
import os.path as osp

import numpy as np
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from .utils import load_pickle, save_pickle, node_counts


class ProteinDataset(InMemoryDataset):
    '''
    Implementation based on https://github.com/KarolisMart/SPECTRE/blob/main/data.py
    '''
    url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/DD"
    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 918
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def raw_file_names(self):
        return ['DD_A.txt', 'DD_graph_indicator.txt', 'DD_graph_labels.txt', 'DD_node_labels.txt']

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        """
        Download raw files.
        """
        for name in ['DD_A.txt', 'DD_graph_indicator.txt', 'DD_graph_labels.txt', 'DD_node_labels.txt']:
            file_path = download_url(f'{self.url}/{name}', self.raw_dir)
            os.rename(file_path, osp.join(self.raw_dir, name))

    def process(self):
        # read
        path = os.path.join(self.root, 'raw')
        data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)

        # split data
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)

        min_num_nodes = 100
        max_num_nodes = 500
        available_graphs = []
        for idx in np.arange(1, data_graph_indicator.max() + 1):
            node_idx = data_graph_indicator == idx
            if node_idx.sum() >= min_num_nodes and node_idx.sum() <= max_num_nodes:
                available_graphs.append(idx)
        available_graphs = torch.Tensor(available_graphs)

        self.num_graphs = len(available_graphs)
        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len

        train_indices, val_indices, test_indices = random_split(available_graphs,
                                                                [train_len, val_len, test_len],
                                                                generator=torch.Generator().manual_seed(1234))
        data_adj = torch.Tensor(np.loadtxt(os.path.join(self.raw_dir, 'DD_A.txt'), delimiter=',')).long() - 1
        data_node_label = torch.Tensor(
            np.loadtxt(os.path.join(self.raw_dir, 'DD_node_labels.txt'), delimiter=',')).long() - 1
        data_graph_indicator = torch.Tensor(
            np.loadtxt(os.path.join(self.raw_dir, 'DD_graph_indicator.txt'), delimiter=',')).long()
        data_graph_types = torch.Tensor(
            np.loadtxt(os.path.join(self.raw_dir, 'DD_graph_labels.txt'), delimiter=',')).long() - 1

        # get information
        self.num_node_type = data_node_label.max() + 1
        self.num_edge_type = 2
        self.num_graph_type = data_graph_types.max() + 1
        print(f"Number of node types: {self.num_node_type}")
        print(f"Number of edge types: {self.num_edge_type}")
        print(f"Number of graph types: {self.num_graph_type}")

        splits = ['train', 'val', 'test']
        indices = [train_indices, val_indices, test_indices]
        counts = {split: 0 for split in splits}
        for i, split in enumerate(splits):
            data_list = []
            ind = indices[i]
            for idx in ind:
                offset = torch.where(data_graph_indicator == idx)[0].min()
                node_idx = data_graph_indicator == idx
                perm = torch.randperm(node_idx.sum()).long()
                reverse_perm = torch.sort(perm)[1]
                nodes = data_node_label[node_idx][perm].long()
                x = torch.zeros_like(nodes).long()
                edge_idx = node_idx[data_adj[:, 0]]
                edge_index = data_adj[edge_idx] - offset
                edge_index[:, 0] = reverse_perm[edge_index[:, 0]]
                edge_index[:, 1] = reverse_perm[edge_index[:, 1]]
                edge_attr = torch.zeros_like(edge_index[:, 0]).long().unsqueeze(-1)
                edge_index, edge_attr = remove_self_loops(edge_index.T, edge_attr)
                data = torch_geometric.data.Data(
                    x=x.unsqueeze(-1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    n_nodes=nodes.shape[0],
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            node_count = node_counts(data_list)
            counts[split] = node_count
            print(f'Number of {split} graphs : {len(data_list)}')
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))


if __name__ == '__main__':
    pass
    # dataset = ProteinDataset('./work/protein/data', 'train')
    # batch = next(iter(dataset))
    # print(batch.x.shape)
    # print(batch.edge_index.shape)
    # print(batch.edge_attr.shape)