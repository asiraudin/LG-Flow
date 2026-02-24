import os.path as osp
from collections import Counter
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from .utils import node_counts, save_pickle, load_pickle


class TPUGraphDataset(InMemoryDataset):
    """
    TPU Graph Dataset for processing TPU Tile graphs into a PyTorch Geometric-compatible format.
    """

    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]
        # if split == 'train':
        #     self.weights = load_pickle(osp.join(self.processed_dir, 'train_weights.pickle'))

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """
        Download raw data files. Taken from LayerDAG
        """

        train_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/train.pth"
        val_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/val.pth"
        test_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/test.pth"

        train_data = torch.load(download_url(train_url, self.raw_dir))
        val_data = torch.load(download_url(val_url, self.raw_dir))
        test_data = torch.load(download_url(test_url, self.raw_dir))

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        """
        Processes raw TPU graph datasets into a PyTorch Geometric-compatible format.
        """
        splits = ['train', 'val', 'test']
        counts = {split: 0 for split in splits}

        for split in splits:
            raw_dataset = torch.load(osp.join(self.raw_dir, f'{split}.pt'))

            data_list = []
            for src, dst, x_n, y in zip(
                raw_dataset["src_list"],
                raw_dataset["dst_list"],
                raw_dataset["x_n_list"],
                raw_dataset["y_list"],
            ):
                edge_index = torch.vstack((src, dst))
                # No edge attributes so setting them all to 1
                edge_attr = torch.zeros(edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
                # x = torch.zeros(x_n.shape[0], 1, dtype=torch.long)
                x = x_n.unsqueeze(-1)
                num_nodes = x.size(0)
                data = torch_geometric.data.Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=num_nodes
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            node_count = node_counts(data_list)
            counts[split] = node_count
            # if split == 'train':
            #     counter = Counter()
            #     for data in data_list:
            #         x = data.x.squeeze().tolist()
            #         counter.update(x)
            #
            #     n_samples = sum(counter.values())
            #     n_classes = 47
            #     class_counts = [value for key, value in sorted(counter.items(), key=lambda item: item[0])]
            #
            #     weights = [n_samples / (n_classes * count) for count in class_counts]
            #     save_pickle(weights, osp.join(self.processed_dir, 'train_weights.pickle'))

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))
