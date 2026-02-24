from typing import Sequence, Any
import tqdm
import torch
import torch.nn.functional as F

import networkx as nx
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.utils import (to_dense_adj, to_dense_batch, cumsum, degree, coalesce, remove_self_loops,
                                   dense_to_sparse, to_networkx, to_undirected)
from torch_geometric.transforms import BaseTransform
import pickle
from collections import Counter
from loguru import logger


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def to_list(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def transform_dataset(dataset, transform):
    data_list = []
    for data in tqdm.tqdm(dataset, miniters=len(dataset) / 50):
        data_list.append(transform(data))
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset._data, dataset.slices = dataset.collate(data_list)
    return dataset


def compute_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    normalized=False,
    normalize=False,
    large_graph=False,
):
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1

    if normalized:
        D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
        I = torch.eye(A.size(0))
        L = I - D12 @ A @ D12
    else:
        D = torch.diag(A.sum(1))
        L = D - A
    eigvals, eigvecs = torch.linalg.eigh(L)

    if large_graph:
        idx1 = torch.argsort(eigvals)[: max_freq // 2]
        idx2 = torch.argsort(eigvals, descending=True)[: max_freq // 2]
        idx = torch.cat([idx1, idx2])
    else:
        idx = torch.argsort(eigvals)[:max_freq]

    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)

    if normalize:
        eignorm: torch.Tensor = eigvecs.norm(p=2, dim=0, keepdim=True)
        eigvecs = eigvecs / eignorm.clamp_min(1e-12).expand_as(eigvecs)

    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)
    return eigvals, eigvecs


def compute_magnetic_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    q=0.25,                 # charge parameter (q=0 -> undirected case)
    normalized=False,       # normalized magnetic Laplacian (MagNet)
    normalize=False,        # L2-normalize eigenvectors (column-wise)
    large_graph=False,      # pick low & high freq halves as in your code
):
    """
    Returns:
        eigvals:  (num_nodes x k) real tensor (k = selected eigen-count)
        eigvecs:  (num_nodes x 2k) real tensor with interleaved [Re, Im, ...]
    """

    # Build (directed) adjacency A \in {0,1}^{n x n}
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0

    # m
    m = (A != A.t()).sum(dim=(-2, -1)) / 2

    min_m_n = torch.where(m > num_nodes, num_nodes, m)

    # safe division
    q = q / torch.where(min_m_n > 1, min_m_n, torch.tensor(1, dtype=min_m_n.dtype))

    # Symmetrized adjacency and antisymmetric phase (MagNet)
    As = ((A + A.t()) > 0).float()                                   # real symmetric
    Theta = 2.0 * torch.pi * q * (A - A.t())                 # real, antisymmetric
    phase = torch.exp(1j * Theta)                            # complex phase
    # Hermitian adjacency: H = As ⊙ exp(i Theta)
    H = As.to(torch.cdouble) * phase                         # Hadamard product

    n = num_nodes

    if normalized:
        # L_N^{(q)} = I - (D_s^{-1/2} As D_s^{-1/2}) ⊙ exp(i Theta)
        ds = As.sum(1)                                       # real degrees of As
        inv_sqrt = (ds.clamp_min(1).pow(-0.5)).to(torch.cdouble)
        Dm = torch.diag(inv_sqrt)
        As_norm = (Dm @ As.to(torch.cdouble) @ Dm)           # real, symmetric
        H_norm = As_norm * phase                             # Hermitian
        L = torch.eye(n, dtype=torch.cdouble) - H_norm
    else:
        # L_U^{(q)} = D_s - H, with D_s = diag(As 1)
        Ds = torch.diag(As.sum(1).to(torch.cdouble))
        L = Ds - H                                           # Hermitian PSD

    # Hermitian eigendecomposition: eigenvalues are real, vectors complex
    eigvals, eigvecs = torch.linalg.eigh(L)                  # shapes: (n,), (n,n)

    # Select frequencies as in your original routine
    if large_graph:
        idx1 = torch.argsort(eigvals.real)[: max_freq // 2]
        idx2 = torch.argsort(eigvals.real, descending=True)[: max_freq // 2]
        idx = torch.cat([idx1, idx2])
    else:
        idx = torch.argsort(eigvals.real)[: max_freq]

    eigvals = eigvals[idx].real.clamp_min(0)                 # (k,)
    eigvecs = eigvecs[:, idx]                                # (n, k) complex

    # Pad when num_nodes < max_freq
    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))

    # Repeat eigvals per node as in your original function
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)      # (n, k_padded)

    return eigvals.float(), eigvecs, q


class Transform:
    def __init__(
        self,
        directed,
        normalized_laplacian,
        normalize_eigenvecs,
        large_graph=False,

    ):
        self.directed = directed
        self.normalized = normalized_laplacian
        self.normalize = normalize_eigenvecs
        self.large_graph = large_graph

    def __call__(self, data):
        if self.directed:
            eigvals, eigvecs, q = compute_magnetic_laplacian_eigen(
                data.edge_index,
                data.num_nodes,
                data.num_nodes,
                0.1,
                self.normalized,
                self.normalize,
                self.large_graph,
            )

            data.eigvals = eigvals
            data.eigvecs_real = eigvecs.real.float()
            data.eigvecs_imag = eigvecs.imag.float()
            data.q = q
        else:
            data.eigvals, data.eigvecs = compute_laplacian_eigen(
                data.edge_index,
                data.num_nodes,
                data.num_nodes,
                self.normalized,
                self.normalize,
                self.large_graph,
            )
        return data


class DegreeTransform(BaseTransform):
    def __call__(self, data):
        idx = data.edge_index[0]
        degrees = degree(idx).unsqueeze(-1)
        data.x = degrees
        return data


def compute_1wl_orbits(g):
    if not nx.is_connected(g):
        logger.info(f'Graph is disconnected !')
        # Get the largest connected component as a subgraph
        largest_cc = max(nx.connected_components(g), key=len)
        subgraph = g.subgraph(largest_cc)

        # Compute its diameter
        diameter = nx.diameter(subgraph)
    else:
        diameter = nx.diameter(g)

    hashes = nx.weisfeiler_lehman_subgraph_hashes(g, iterations=diameter)
    final_hashes = {key: value[-1] for key, value in hashes.items()}

    list_hashes = list(final_hashes.values())
    orbits = {}
    for i, hash_value in enumerate(list_hashes):
        if hash_value not in orbits:
            orbits[hash_value] = []
        orbits[hash_value].append(i)

    non_trivial_orbits = [value for key, value in orbits.items() if len(value) > 1]
    return non_trivial_orbits


class OrbitTransform(BaseTransform):
    def __call__(self, data):
        g = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True
        )
        non_trivial_orbits = compute_1wl_orbits(g)
        orbits_attr = torch.zeros_like(data.x[..., [0]])

        for i, idxs in enumerate(non_trivial_orbits):
            orbits_attr[idxs] = i + 1
        data.orbits = orbits_attr
        return data
