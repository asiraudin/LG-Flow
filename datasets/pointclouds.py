import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import from_networkx

from .utils import load_pickle, save_pickle, node_counts


class PointCloudsDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 41
        self.n_nodes = 5037
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def raw_file_names(self):
        return ["point_cloud.pkl"]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        url = 'https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/point_cloud.pkl'
        download_url(
            url, osp.join(self.raw_dir)
        )

    def process(self):
        splits = ['train', 'val', 'test']
        counts = {split: 0 for split in splits}
        raw_dataset = load_pickle(os.path.join(self.raw_dir, f"point_cloud.pkl"))
        for split in splits:
            raw_graphs = raw_dataset[split]
            data_list = []
            for idx, graph in enumerate(raw_graphs):
                data = from_networkx(graph)
                n = graph.number_of_nodes()
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

            node_count = node_counts(data_list)
            counts[split] = node_count
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))