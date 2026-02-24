import os.path as osp
import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from .utils import node_counts, save_pickle, load_pickle


class PriceDataset(InMemoryDataset):
    """
    Class for generating and processing synthetic graph data.
    """

    def __init__(
        self,
        root,
        split,
        graph_type,
        w_min=0.5,
        w_max=0.5,
        acyclic=True,
        transform=None,
        pre_transform=None,
    ):
        self.num_graphs = 8704
        self.p_threshold = 0.6
        self.graph_type = graph_type
        self.w_min = w_min
        self.w_max = w_max
        self.acyclic = acyclic
        super().__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        """
        Generate synthetic data and convert to PyTorch Geometric format.
        """
        data_list = []

        for _ in range(self.num_graphs):
            adj = self.generate_structure(
                self.p_threshold,
                self.graph_type,
                self.w_min,
                self.w_max,
                self.acyclic,
            )
            edge_index = torch.tensor(np.vstack(np.nonzero(adj)), dtype=torch.long)
            edge_attr = torch.zeros(edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
            n = adj.shape[-1]
            x = torch.zeros(n, 1, dtype=torch.long)

            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr
            )
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

    @staticmethod
    def generate_structure(
        p_threshold, graph_type, w_min, w_max, acyclic
    ):
        """
        Generate a synthetic graph structure as an adjacency matrix.

        Parameters:
        - degree: Average degree of nodes.
        - graph_type: Type of graph to generate ('erdos-renyi', 'barabasi-albert', 'full').
        - w_min, w_max: Minimum and maximum absolute edge weights.
        - acyclic: If True, ensures the generated graph is acyclic (DAG).

        Returns:
        - adj_matrix: The adjacency matrix of the generated graph.
        """

        w_min, w_max = abs(w_min), abs(w_max)
        if w_min > w_max:
            raise ValueError(
                f"Minimum weight must be <= maximum weight: {w_min} > {w_max}"
            )

        elif graph_type == "ba":  # barabasi-albert
            # Known as Price's model in network theory when acyclic
            # Dynamically setting m to allow graphs with different number of nodes
            if not acyclic:
                raise ValueError("Barabasi-Albert model only generates acyclic graphs")

            m = max(1, int(round(np.log2(64))))  # m = 6 for 64 nodes
            # m = int(round(degree / 2))
            edge_flags = np.zeros([64, 64])
            bag = [0]
            for i in range(1, 64):
                dest = np.random.choice(bag, size=m)
                for j in dest:
                    edge_flags[i, j] = 1
                bag.append(i)
                bag.extend(dest)
        else:
            raise ValueError(f"Unknown graph type {graph_type}")

        # Randomly permute edges
        perms = np.random.permutation(np.eye(64, 64))
        edge_flags = perms.T @ edge_flags @ perms

        # Generate random edge weights (optional)
        # edge_weights = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes])
        # edge_weights[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

        adj_matrix = edge_flags  # Weighting removed for simplicity; can be added back
        return adj_matrix
