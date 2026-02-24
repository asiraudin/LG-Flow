import os.path as osp
import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from .utils import node_counts, save_pickle, load_pickle


class ERDAGDataset(InMemoryDataset):
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

        train_data_list = data_list[:1280]
        val_data_list = data_list[1280:1600]
        test_data_list = data_list[1600:]
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

        if graph_type == "er":  # erdos-renyi
            # p_threshold = float(degree) / (num_nodes - 1)
            num_nodes = np.random.randint(20, 80)
            p_edge = (np.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
            # Set diagonal to zero
            np.fill_diagonal(p_edge, 0)
            edge_flags = (
                np.tril(p_edge, k=-1) if acyclic else p_edge
            )  # Force DAG if acyclic=True
        else:
            raise ValueError(f"Unknown graph type {graph_type}")

        # Randomly permute edges
        perms = np.random.permutation(np.eye(num_nodes, num_nodes))
        edge_flags = perms.T @ edge_flags @ perms

        # Generate random edge weights (optional)
        # edge_weights = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes])
        # edge_weights[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

        adj_matrix = edge_flags  # Weighting removed for simplicity; can be added back
        return adj_matrix
