from typing import Sequence, Any
import time
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

LAPLACIAN_TIME_ATTR = "_laplacian_preprocess_time_s"
ORBIT_TIME_ATTR = "_orbit_preprocess_time_s"


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def init_preprocess_time_totals() -> dict[str, float]:
    return {"laplacian_time_s": 0.0, "orbit_time_s": 0.0}


def pop_preprocess_times(data: Data) -> dict[str, float]:
    laplacian_time_s = float(getattr(data, LAPLACIAN_TIME_ATTR, 0.0))
    orbit_time_s = float(getattr(data, ORBIT_TIME_ATTR, 0.0))
    if hasattr(data, LAPLACIAN_TIME_ATTR):
        delattr(data, LAPLACIAN_TIME_ATTR)
    if hasattr(data, ORBIT_TIME_ATTR):
        delattr(data, ORBIT_TIME_ATTR)
    return {
        "laplacian_time_s": laplacian_time_s,
        "orbit_time_s": orbit_time_s,
    }


def consume_preprocess_times(data: Data, totals: dict[str, float]) -> None:
    times = pop_preprocess_times(data)
    totals["laplacian_time_s"] += times["laplacian_time_s"]
    totals["orbit_time_s"] += times["orbit_time_s"]


def log_preprocess_times(dataset_name: str, totals: dict[str, float]) -> None:
    logger.info(
        f"{dataset_name} preprocessing times | laplacian: {totals['laplacian_time_s']:.2f}s | "
        f"orbits: {totals['orbit_time_s']:.2f}s"
    )


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
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float64)  # (e,)
    row, col = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, dtype=torch.float64)  # (n,)
    degree.scatter_add_(0, row, edge_weight)

    if large_graph:
        if normalized:
            inv_sqrt_degree = degree.clamp_min(1).pow(-0.5)  # (n,)
            laplacian_values = -edge_weight * inv_sqrt_degree[row] * inv_sqrt_degree[col]  # (e,)
            diagonal_values = torch.ones(num_nodes, dtype=torch.float64)  # (n,)
        else:
            laplacian_values = -edge_weight  # (e,)
            diagonal_values = degree  # (n,)

        diagonal_index = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)  # (2, n)
        laplacian_index = torch.cat((edge_index, diagonal_index), dim=1)  # (2, e + n)
        laplacian_values = torch.cat((laplacian_values, diagonal_values), dim=0)  # (e + n,)
        L = torch.sparse_coo_tensor(
            laplacian_index,
            laplacian_values,
            (num_nodes, num_nodes),
        ).coalesce()

        k = min(max_freq, num_nodes)  # ()
        if num_nodes >= 3 * k:
            eigvals, eigvecs = torch.lobpcg(
                L,
                k=k,
                largest=False,
                method="ortho",
            )
        else:
            L_dense = L.to_dense()  # (n, n)
            eigvals, eigvecs = torch.linalg.eigh(L_dense)
    else:
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)  # (n, n)
        A[edge_index[0], edge_index[1]] = 1

        if normalized:
            D12 = torch.diag(A.sum(1).clip(1) ** -0.5)  # (n, n)
            I = torch.eye(A.size(0), dtype=torch.float64)  # (n, n)
            L = I - D12 @ A @ D12  # (n, n)
        else:
            D = torch.diag(A.sum(1))  # (n, n)
            L = D - A  # (n, n)
        eigvals, eigvecs = torch.linalg.eigh(L)

    idx = torch.argsort(eigvals)[:max_freq]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)

    if normalize:
        eignorm: torch.Tensor = eigvecs.norm(p=2, dim=0, keepdim=True)
        eigvecs = eigvecs / eignorm.clamp_min(1e-12).expand_as(eigvecs)

    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)  # (n, k)
    return eigvals.float(), eigvecs.float()


def compute_magnetic_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    q=0.25,                 # charge parameter (q=0 -> undirected case)
    normalized=False,       # normalized magnetic Laplacian (MagNet)
    normalize=False,        # L2-normalize eigenvectors (column-wise)
    large_graph=False,
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
        num_vecs,
        large_graph=False,

    ):
        self.directed = directed
        self.normalized = normalized_laplacian
        self.normalize = normalize_eigenvecs
        self.num_vecs = num_vecs
        self.large_graph = large_graph

    def __call__(self, data):
        start_time = time.perf_counter()
        # Keep a fixed spectral width across graphs so variable-size datasets collate cleanly.
        max_freq = self.num_vecs
        if self.directed:
            eigvals, eigvecs, q = compute_magnetic_laplacian_eigen(
                data.edge_index,
                data.num_nodes,
                max_freq,
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
                max_freq,
                self.normalized,
                self.normalize,
                self.large_graph,
            )
        setattr(data, LAPLACIAN_TIME_ATTR, time.perf_counter() - start_time)
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
        start_time = time.perf_counter()
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
        setattr(data, ORBIT_TIME_ATTR, time.perf_counter() - start_time)
        return data
