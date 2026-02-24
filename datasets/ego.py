import os
import os.path as osp
import pickle as pkl

import torch
from torch_geometric.utils import dense_to_sparse, from_networkx
from torch_geometric.data import InMemoryDataset, download_url, Data

from .utils import node_counts, save_pickle, load_pickle


class EgoDataset(InMemoryDataset):
    url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 757
        super().__init__(root, transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def raw_file_names(self):
        return ['data.pkl']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        os.rename(file_path, osp.join(self.raw_dir, 'data.pkl'))

    def process(self):
        networks = pkl.load(open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb'))
        data_list = []
        for network in networks:
            data = from_networkx(network)
            n = network.number_of_nodes()
            x = torch.zeros(n, 1, dtype=torch.long)
            edge_attr = torch.zeros(data.edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data.x = x
            data.edge_attr = edge_attr
            data.num_nodes = n_nodes

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        indices = torch.randperm(self.n_graphs, generator=g_cpu)
        test_len = int(round(self.n_graphs * 0.2))
        train_len = int(round(self.n_graphs * 0.8))
        val_len = int(round(self.n_graphs * 0.2))
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        train_indices = indices[:train_len]
        val_indices = indices[:val_len]
        test_indices = indices[train_len:]

        train_data_list = [data_list[idx] for idx in train_indices]
        val_data_list = [data_list[idx] for idx in val_indices]
        test_data_list = [data_list[idx] for idx in test_indices]
        counts = {'train': 0, 'val': 0, 'test': 0}
        for split, data_list in zip(['train', 'val', 'test'], [train_data_list, val_data_list, test_data_list]):
            node_count = node_counts(data_list)
            counts[split] = node_count

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))


if __name__ == '__main__':
    dataset = EgoDataset('./work/ego/data', 'train')

