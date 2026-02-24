import torch.nn as nn
import numpy as np
import networkx as nx
import concurrent.futures
from functools import partial
from tqdm import tqdm

import pygsp as pg
from scipy.linalg import eigvalsh
from scipy.stats import chi2, ks_2samp, powerlaw
from .dist_helper import (
    compute_mmd,
    gaussian_emd,
    gaussian_tv,
)
from torch_geometric.utils import to_networkx

import wandb
import time


############################ Distributional measures ############################

# Degree distribution -----------------------------------------------------------


def degree_worker(G, is_out=True):
    if is_out:
        histogram = [value for _, value in G.in_degree]
    else:
        histogram = [value for _, value in G.out_degree]
    return np.array(histogram)


def degree_stats(
        graph_ref_list, graph_pred_list, is_parallel=True, is_out=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    # prev = datetime.now()
    if is_parallel:
        degree_worker_partial = partial(degree_worker, is_out=is_out)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker_partial, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(
                    degree_worker_partial, graph_pred_list_remove_empty
            ):
                sample_pred.append(deg_hist)
    else:
        attribute = "out_degree" if is_out else "in_degree"
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(
                [value for _, value in getattr(graph_ref_list[i], attribute)]
            )
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                [
                    value
                    for _, value in getattr(graph_pred_list_remove_empty[i], attribute)
                ]
            )
            sample_pred.append(degree_temp)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


# Cluster coefficient -----------------------------------------------------------


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
        graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    # prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# Spectre -----------------------------------------------------------------------


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(
            np.asarray(nx.directed_laplacian_matrix(G, walk_type="pagerank"))
        )
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1: n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(
        graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    # prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                    spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                    spectral_worker,
                    graph_pred_list_remove_empty,
                    [n_eigvals for i in graph_ref_list],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


# Wavelet -----------------------------------------------------------------------


def eigh_worker(G):
    L = np.asarray(nx.directed_laplacian_matrix(G, walk_type="pagerank"))
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop ** 2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
        eigvec_ref_list,
        eigval_ref_list,
        eigvec_pred_list,
        eigval_pred_list,
        is_parallel=False,
        compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """

    # prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                    get_spectral_filter_worker,
                    eigvec_ref_list,
                    eigval_ref_list,
                    [filters for i in range(len(eigval_ref_list))],
                    [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                    get_spectral_filter_worker,
                    eigvec_pred_list,
                    eigval_pred_list,
                    [filters for i in range(len(eigval_pred_list))],
                    [bound for i in range(len(eigval_pred_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


############################ Validity measures ############################
def eval_acc_directed_acyclic_graph(G_list):
    count = 0
    for gg in tqdm(G_list):
        if nx.is_directed_acyclic_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_dag_and_barabasi_albert(G):
    return nx.is_directed_acyclic_graph(G) and is_barabasi_albert(G)


def is_barabasi_albert(G, strict=True):
    """
    Check if a given graph follows a Barabási–Albert (BA) model.

    Parameters:
    - G: NetworkX graph (assumed undirected)
    - strict: If True, return a boolean decision (p > 0.9).
              If False, return the p-value from Kolmogorov-Smirnov 2 samples test.

    Returns:
    - True if the graph follows a BA model (if strict=True)
    - p-value of the Kolmogorov-Smirnov 2 samples test (if strict=False)
    """
    degrees = np.array([d for _, d in G.degree()])

    # Estimate power-law exponent
    degrees = degrees[degrees > 1]  # ignoring degree=1 nodes
    if len(degrees) < 2:
        return False

    # Verify m < n in the graph
    if len(G) <= 6:
        return False

    # Generate a synthetic BA graph with the same number of nodes and m
    G_synthetic = nx.barabasi_albert_graph(n=len(G), m=6)
    synthetic_degrees = np.array([d for _, d in G_synthetic.degree()])

    # Perform Kolmogorov-Smirnov test
    p_value = 1 - ks_2samp(degrees, synthetic_degrees)[1]

    return p_value > 0.9 if strict else p_value


def eval_barabasi_albert(G_list):
    count = 0
    for G in tqdm(G_list):
        if is_barabasi_albert(G):
            count += 1
    return count / float(len(G_list))


def eval_acc_dag_and_barabasi_albert(G_list):
    count = 0
    for gg in tqdm(G_list):
        if is_dag_and_barabasi_albert(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_scene_graph(G_list, val_scene_graph_fn):
    count = 0
    for gg in tqdm(G_list):
        if val_scene_graph_fn(gg):
            count += 1
    return count / float(len(G_list))


def eval_connected_graph(G_list):
    count = 0
    for gg in tqdm(G_list):
        adj = nx.adjacency_matrix(gg).toarray()
        adj = np.maximum(adj, adj.T)
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
            count += 1
    return count / float(len(G_list))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in tqdm(fake_graphs):
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def time_eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    count_non_validated = 0
    for fake_g in tqdm(fake_graphs):
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                timeout, isomorphic = is_isomorphic_with_timeout(fake_g, train_g)
                if isomorphic:
                    count += 1
                    break
                elif timeout:
                    count_non_validated += 1
    return count / float(len(fake_graphs)), count_non_validated / float(
        len(fake_graphs)
    )


def eval_fraction_unique_non_isomorphic_valid(
        fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    start = time.time()
    for i, fake_g in enumerate(tqdm(fake_graphs)):
        if i % 100 == 0:
            print(f"Processing graph {i}")
            print(f"Time elapsed: {time.time() - start}")
        unique = True
        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
                                         float(len(fake_graphs)) - count_non_unique - count_isomorphic
                                 ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


import threading


def is_isomorphic_worker(fake_g, train_g, result_container):
    """Worker function to check isomorphism and store the result."""
    is_attributed = "label" in train_g.nodes[0]
    node_match_fn = lambda x, y: x["label"] == y["label"] if is_attributed else None
    result_container.append(
        # nx.is_isomorphic(
        #     fake_g, train_g
        # )
        nx.is_isomorphic(
            fake_g,
            train_g,
            node_match=node_match_fn,
        )
    )  # Store result in a shared list


def is_isomorphic_with_timeout(fake_g, train_g, timeout=5):
    """Check if two graphs are isomorphic with a timeout (fixed thread-based)."""
    result_container = []  # Shared list to store the result
    thread = threading.Thread(
        target=is_isomorphic_worker,
        args=(fake_g, train_g, result_container),
        daemon=True,
    )
    thread.start()

    thread.join(timeout)  # Wait for the thread to finish within timeout

    if thread.is_alive():
        print("is_isomorphic took too long!")
        return True, False  # Timeout occurred

    return False, (
        result_container[0] if result_container else False
    )  # Return actual result if available


def time_eval_fraction_unique_non_isomorphic_valid(
        fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    count_non_unique_non_validated = 0
    count_isomorphic_non_validated = 0
    fake_evaluated = []
    start = time.time()
    for i, fake_g in enumerate(tqdm(fake_graphs)):
        if i % 100 == 0:
            print(f"Processing graph {i}")
            print(f"Time elapsed: {time.time() - start}")
        unique = True
        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                timeout, isomorphic = is_isomorphic_with_timeout(fake_g, fake_old)
                if isomorphic:
                    count_non_unique += 1
                    unique = False
                    break
                elif timeout:
                    count_non_unique_non_validated += 1
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    timeout, isomorphic = is_isomorphic_with_timeout(fake_g, train_g)
                    if isomorphic:
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
                    elif timeout:
                        count_isomorphic_non_validated += 1
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
                                         float(len(fake_graphs)) - count_non_unique - count_isomorphic
                                 ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    frac_non_unique_non_validated = count_non_unique_non_validated / float(
        len(fake_graphs)
    )  # Fraction of graphs non-validated due to timeout
    frac_isomorphic_non_validated = count_non_unique_non_validated / float(
        len(fake_graphs)
    )  # Fraction of graphs non-validated due to timeout
    return (
        frac_unique,
        frac_unique_non_isomorphic,
        frac_unique_non_isomorphic_valid,
        frac_non_unique_non_validated,
        frac_isomorphic_non_validated,
    )


def compute_ratios(gen_metrics, ref_metrics, metrics_keys):
    print("Computing ratios of metrics: ", metrics_keys)
    if ref_metrics is not None and len(metrics_keys) > 0:
        ratios = {}
        for key in metrics_keys:
            try:
                ref_metric = round(ref_metrics[key], 4)
            except:
                print(key, "not found")
                continue
            if ref_metric != 0.0:
                ratios["ratio/" + key + "_ratio"] = gen_metrics[key] / ref_metric
            else:
                print(f"WARNING: Reference {key} is 0. Skipping its ratio.")
        if len(ratios) > 0:
            ratios["ratio/average_ratio"] = sum(ratios.values()) / len(ratios)
        else:
            ratios["ratio/average_ratio"] = -1
            print(f"WARNING: no ratio being saved.")
    else:
        print("WARNING: No reference metrics for ratio computation.")
        ratios = {}
    return ratios


############################ Metrics classes ############################
class DirectedSamplingMetrics(nn.Module):
    def __init__(self, dataloaders, metrics_list, graph_type, compute_emd, root=None, test=False):
        super().__init__()

        # self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        # self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        # self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.train_digraphs = self.loader_to_nx(
            dataloaders["train"], directed=True
        )
        self.val_digraphs = self.loader_to_nx(
            dataloaders["val"], directed=True
        )
        self.test_digraphs = self.loader_to_nx(
            dataloaders["test"], directed=True
        )

        self.reference_digraphs = self.test_digraphs if test else self.val_digraphs
        self.metrics_prefix = "test" if test else "sampling"
        self.num_graphs_test = len(self.test_digraphs)
        self.num_graphs_val = len(self.val_digraphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list
        self.graph_type = graph_type

        # Store for wavelet computation
        self.val_ref_eigvals, self.val_ref_eigvecs = compute_list_eigh(
            self.val_digraphs
        )
        self.test_ref_eigvals, self.test_ref_eigvecs = compute_list_eigh(
            self.test_digraphs
        )
        self.ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
        self.ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals
        self.root = root

        # self.compute_train_ref_mmd()

    def compute_train_ref_mmd(self):
        self.ref_metrics = {}
        if "out_degree" in self.metrics_list:
            print("Pre-computing degree stats..")
            self.train_out_degree = degree_stats(
                self.reference_digraphs,
                self.train_digraphs,
                is_parallel=True,
                is_out=True,
                compute_emd=self.compute_emd,
            )
            self.ref_metrics[f"{self.metrics_prefix}/out_degree"] = self.train_out_degree

        if "in_degree" in self.metrics_list:
            print("Pre-computing degree stats..")
            self.train_in_degree = degree_stats(
                self.reference_digraphs,
                self.train_digraphs,
                is_parallel=True,
                is_out=False,
                compute_emd=self.compute_emd,
            )
            self.ref_metrics[f"{self.metrics_prefix}/in_degree"] = self.train_in_degree

        if "spectre" in self.metrics_list:
            print("Pre-computing spectre stats...")
            self.train_spectre = spectral_stats(
                self.reference_digraphs,
                self.train_digraphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )
            self.ref_metrics[f"{self.metrics_prefix}/spectre"] = self.train_spectre


        if "clustering" in self.metrics_list:
            print("Pre-computing clustering stats...")
            self.train_clustering = clustering_stats(
                self.reference_digraphs,
                self.train_digraphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            self.ref_metrics[f"{self.metrics_prefix}/clustering"] = self.train_clustering


        if "wavelet" in self.metrics_list:
            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(
                self.train_digraphs
            )
            self.train_wavelet = spectral_filter_stats(
                eigvec_ref_list=self.ref_eigvecs,
                eigval_ref_list=self.ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            self.ref_metrics[f"{self.metrics_prefix}/wavelet"] = self.train_wavelet


    def loader_to_nx(self, loader, directed=False):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                if directed:
                    networkx_graphs.append(
                        to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=False,
                            remove_self_loops=True,
                        )
                    )
                else:
                    networkx_graphs.append(
                        to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=True,
                            remove_self_loops=True,
                        )
                    )
        return networkx_graphs

    def is_scene_graph(self, G):
        pass

    def forward(
            self,
            generated_graphs: list,
            ref_metrics=None,
            local_rank=0,
            test=False,
    ):
        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(self.reference_digraphs)}"
            )
        networkx_digraphs = []
        # networkx_graphs = []
        adjacency_matrices = []
        if local_rank == 0:
            print("Building networkx graphs...")

        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            adjacency_matrices.append(A)

            nx_digraph = nx.from_numpy_array(
                A, create_using=nx.DiGraph
            )  # we need to specify it is directed
            # nx_graph = nx.from_numpy_array(A, create_using=nx.Graph)

            # need to add labels if it's a scene graph
            if self.graph_type in ["visual_genome", "tpu_tile"]:
                for i, node in enumerate(nx_digraph.nodes()):
                    nx_digraph.nodes[i]["label"] = node_types[i].item()

            networkx_digraphs.append(nx_digraph)
            # networkx_graphs.append(nx_graph)
        if test:
            np.savez(self.root + "/generated_adjs.npz", *adjacency_matrices)

        to_log = {}

        if "out_degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing out-degree stats...")
            out_degree = degree_stats(
                self.reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                is_out=True,
                compute_emd=self.compute_emd,
            )
            to_log[f"{self.metrics_prefix}/out_degree"] = out_degree
            # if wandb.run:
            #     wandb.run.summary['out_degree'] = out_degree

        if "in_degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing in-degree stats...")
            in_degree = degree_stats(
                self.reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                is_out=False,
                compute_emd=self.compute_emd,
            )
            to_log[f"{self.metrics_prefix}/in_degree"] = in_degree
            # if wandb.run:
            #     wandb.run.summary['in_degree'] = in_degree

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                self.reference_digraphs,
                networkx_digraphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log[f"{self.metrics_prefix}/clustering"] = clustering
            # if wandb.run:
            #     wandb.run.summary['clustering'] = clustering

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                self.reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log[f"{self.metrics_prefix}/spectre"] = spectre
            # if wandb.run:
            #   wandb.run.summary['spectre'] = spectre

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
            ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(
                networkx_digraphs
            )
            wavelet = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log[f"{self.metrics_prefix}/wavelet"] = wavelet
            # if wandb.run:
            #     wandb.run.summary["wavelet"] = wavelet

        if "connected" in self.metrics_list:
            if local_rank == 0:
                print("Computing connected accuracy...")
            con_acc = eval_connected_graph(networkx_digraphs)
            to_log[f"{self.metrics_prefix}/con_acc"] = con_acc
            # if wandb.run:
            #     wandb.run.summary['con_acc'] = con_acc

        if "ba" in self.metrics_list:
            if local_rank == 0:
                print("Computing BA accuracy...")
            ba_acc = eval_barabasi_albert(networkx_digraphs)
            to_log[f"{self.metrics_prefix}/ba_acc"] = ba_acc

        if "dag" in self.metrics_list:
            if local_rank == 0:
                print("Computing DAG accuracy...")
            dag_acc = eval_acc_directed_acyclic_graph(networkx_digraphs)
            to_log[f"{self.metrics_prefix}/dag_acc"] = dag_acc
            if "ba" in self.metrics_list:
                dag_ba_acc = eval_acc_dag_and_barabasi_albert(networkx_digraphs)
                to_log[f"{self.metrics_prefix}/dag_ba_acc"] = dag_ba_acc

        if "scene_graph" in self.metrics_list:
            if local_rank == 0:
                print("Computing scene graph accuracy...")
            scene_graph_acc = eval_acc_scene_graph(
                networkx_digraphs, val_scene_graph_fn=self.is_scene_graph
            )
            to_log[f"{self.metrics_prefix}/scene_graph_acc"] = scene_graph_acc
            # if wandb.run:
            #     wandb.run.summary['scene_graph_acc'] = scene_graph_acc

        if "valid" in self.metrics_list:
            validity_dictionary = {
                "tpu_tile": nx.is_directed_acyclic_graph,
                "visual_genome": self.is_scene_graph,
            }
            validity_metric = validity_dictionary[self.graph_type]

            if local_rank == 0:
                print("Computing all fractions...")
            (
                frac_unique,
                frac_unique_non_isomorphic,
                frac_unique_non_isomorphic_valid,
                frac_non_unique_non_validated,
                frac_isomorphic_non_validated,
            ) = time_eval_fraction_unique_non_isomorphic_valid(
                networkx_digraphs, self.train_digraphs, validity_func=validity_metric
            )
            frac_isomorphic, frac_isomorphic_non_validated2 = (
                time_eval_fraction_isomorphic(networkx_digraphs, self.train_digraphs)
            )
            frac_non_isomorphic = 1.0 - frac_isomorphic
            to_log.update(
                {
                    f"{self.metrics_prefix}/frac_unique": frac_unique,
                    f"{self.metrics_prefix}/frac_unique_non_iso": frac_unique_non_isomorphic,
                    f"{self.metrics_prefix}/frac_unic_non_iso_valid": frac_unique_non_isomorphic_valid,
                    f"{self.metrics_prefix}/frac_non_iso": frac_non_isomorphic,
                    f"{self.metrics_prefix}/frac_non_unique_non_validated": frac_non_unique_non_validated,
                    f"{self.metrics_prefix}/frac_isomorphic_non_validated": frac_isomorphic_non_validated,
                    f"{self.metrics_prefix}/frac_isomorphic_non_validated2": frac_isomorphic_non_validated2,
                }
            )

        # ratios = compute_ratios(
        #     gen_metrics=to_log,
        #     ref_metrics=self.ref_metrics,
        #     metrics_keys=[
        #                 f"{self.metrics_prefix}/out_degree",
        #                 f"{self.metrics_prefix}/in_degree",
        #                 f"{self.metrics_prefix}/clustering",
        #                 f"{self.metrics_prefix}/spectre",
        #                 f"{self.metrics_prefix}/wavelet",
        #             ],
        #         )
        # to_log.update(ratios)
        # print('Ratios computed')
        if wandb.run:
            wandb.log(to_log, commit=False)
        print('Metrics logged')
        return to_log

    def reset(self):
        pass


# Override loader so that the node labels are kept
def node_attributed_loader_to_nx(loader, directed=False):
    networkx_graphs = []
    for i, batch in enumerate(loader):
        data_list = batch.to_data_list()
        for j, data in enumerate(data_list):
            if directed:
                labels = data.x.squeeze()
                new_nx_graph = to_networkx(
                    data,
                    node_attrs=None,
                    edge_attrs=None,
                    to_undirected=False,
                    remove_self_loops=True,
                )
                # add label to the nodes
                for i, node in enumerate(new_nx_graph.nodes()):
                    new_nx_graph.nodes[i]["label"] = labels[i].item()
                networkx_graphs.append(new_nx_graph)
            else:
                raise ValueError("Graphs must be directed, please set directed=True")

    return networkx_graphs


class TPUSamplingMetrics(DirectedSamplingMetrics):
    def __init__(self, dataloaders, root=None, test=False):
        super().__init__(
            dataloaders=dataloaders,
            metrics_list=[
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "connected",
                "dag",
                "valid",
                "unique",
            ],
            graph_type="tpu_tile",
            compute_emd=False,
            root=root,
            test=test
        )

    # Override loader so that the node labels are kept
    def loader_to_nx(self, loader, directed=False):
        return node_attributed_loader_to_nx(loader, directed=directed)


class PriceSamplingMetrics(DirectedSamplingMetrics):
    def __init__(self, dataloaders, root=None, test=False):
        super().__init__(
            dataloaders=dataloaders,
            metrics_list=[
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "connected",
                "dag",
                "valid",
                "unique",
                "ba"
            ],
            graph_type="tpu_tile",
            compute_emd=False,
            root=root,
            test=test
        )
