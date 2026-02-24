import torch
from loguru import logger


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p


def compute_n_nodes_distr(train_n_nodes, val_n_nodes, test_n_nodes):
    max_n_nodes = max(
        max(train_n_nodes), max(val_n_nodes), max(test_n_nodes)
    )
    min_n_nodes = min(
        min(train_n_nodes), min(val_n_nodes), min(test_n_nodes)
    )
    total_n_nodes = train_n_nodes + val_n_nodes + test_n_nodes
    average_n_nodes = sum(n * count for n, count in total_n_nodes.items()) / sum(total_n_nodes.values())
    logger.info(f'Number of nodes : max = {max_n_nodes} - min = {min_n_nodes} - avg = {average_n_nodes}')
    n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
    for key, value in total_n_nodes.items():
        n_nodes[key] += value
    n_nodes = n_nodes / n_nodes.sum()

    nodes_dist = DistributionNodes(n_nodes)
    return nodes_dist