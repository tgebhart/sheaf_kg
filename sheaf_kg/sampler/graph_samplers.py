from typing import Iterable, List

import torch
from pykeen.triples.instances import SubGraphSLCWAInstances, BaseBatchedSLCWAInstances
from pykeen.triples.utils import compute_compressed_adjacency_list


class LinearSubGraphSLCWAInstances(BaseBatchedSLCWAInstances):
    """Pre-batched training instances for SLCWA of linear chains."""

    def __init__(self, **kwargs):
        """
        Initialize the instances.

        :param kwargs:
            keyword-based parameters passed to :meth:`BaseBatchedSLCWAInstances.__init__`
        """
        super().__init__(**kwargs)
        # indexing
        self.degrees, self.offset, self.neighbors = compute_compressed_adjacency_list(
            mapped_triples=self.mapped_triples
        )

    def subgraph_sample(self) -> List[int]:
        """sample a single linear chain with disjoint vertices and self.batch_size edges"""
        node_weights = self.degrees.detach().clone()
        edge_picked = torch.zeros(self.mapped_triples.shape[0], dtype=torch.bool)
        node_picked = torch.zeros(self.degrees.shape[0], dtype=torch.bool)

        result = []
        # choose a random vertex to start
        current_head = torch.randint(node_picked.numel(), size=tuple())
        node_picked[current_head] = True
        for _ in range(self.batch_size):
            # sample the initial node

            start = self.offset[current_head]
            current_head_degree = self.degrees[current_head].item()
            # probably need to have something to do when the degree is zero
            # or backtracking when there just isn't a path of sufficient length?
            stop = start + current_head_degree
            adj_list = self.neighbors[start:stop, :]

            chosen_edge_index = torch.randint(current_head_degree, size=tuple())
            chosen_edge = adj_list[chosen_edge_index]
            edge_number = chosen_edge[0]

            # find an edge which has not been picked and whose tail vertex also has not been picked
            while edge_picked[edge_number] and node_picked[chosen_edge[1]]:
                chosen_edge_index = torch.randint(current_head_degree, size=tuple())
                chosen_edge = adj_list[chosen_edge_index]
                edge_number = chosen_edge[0]
            result.append(edge_number.item())

            edge_picked[edge_number] = True

            current_head = chosen_edge[1]

        return result

    # docstr-coverage: inherited
    def iter_triple_ids(self) -> Iterable[List[int]]:  # noqa: D102
        yield from (self.subgraph_sample() for _ in self.split_workload(n=len(self)))
