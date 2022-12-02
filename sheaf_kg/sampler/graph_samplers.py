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
        # edge_picked = torch.zeros(self.mapped_triples.shape[0], dtype=torch.bool)
        # node_picked = torch.zeros(self.degrees.shape[0], dtype=torch.bool)

        redo = True

        while redo:
            edge_picked = torch.zeros(self.mapped_triples.shape[0], dtype=torch.bool)
            node_picked = torch.zeros(self.degrees.shape[0], dtype=torch.bool)
            redo = False
            result = []
            # choose a random vertex to start
            current_head = torch.randint(node_picked.numel(), size=tuple())

            while node_weights[current_head] == 0:
                current_head = torch.randint(node_picked.numel(), size=tuple())

            node_picked[current_head] = True
            for _ in range(self.batch_size):
                start = self.offset[current_head]
                current_head_degree = self.degrees[current_head].item()
                if current_head_degree == 0:
                    redo = True
                    # start over
                    break

                stop = start + current_head_degree
                adj_list = self.neighbors[start:stop, :]

                # check that there is at least one valid edge to pick.
                invalid_edges = edge_picked[adj_list[:,0]] | node_picked[adj_list[:,1]]
                # probably node_picked is sufficient, huh?
                if torch.all(invalid_edges):
                    redo = True
                    break

                chosen_edge_index = torch.randint(current_head_degree, size=tuple())
                chosen_edge = adj_list[chosen_edge_index]
                edge_number = chosen_edge[0]
                # find an edge which has not been picked and whose tail vertex also has not been picked
                while invalid_edges[chosen_edge_index]:
                    chosen_edge_index = torch.randint(current_head_degree, size=tuple())
                    chosen_edge = adj_list[chosen_edge_index]
                    edge_number = chosen_edge[0]

                result.append(edge_number.item())

                edge_picked[edge_number] = True
                current_head = chosen_edge[1]
                node_picked[current_head] = True

        return result

    # docstr-coverage: inherited
    def iter_triple_ids(self) -> Iterable[List[int]]:  # noqa: D102
        yield from (self.subgraph_sample() for _ in self.split_workload(n=len(self)))
