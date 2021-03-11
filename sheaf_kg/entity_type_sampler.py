# -*- coding: utf-8 -*-

from typing import Collection, Optional, Tuple

import torch

from .negative_sampler import NegativeSampler
from ..triples import TriplesFactory

__all__ = [
    'EntityTypeSampler',
]

LOOKUP = {'h': 0, 'r': 1, 't': 2}

class EntityTypeSampler(NegativeSampler):
    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, q=10),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        entity_types: list,
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = False
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param entity_types: A list of entity type names as strings
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        """
        super().__init__(
            triples_factory=triples_factory,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
        )
        self.corruption_scheme = ('h', 't')
        self.entity_types = entity_types
        self.type_map = self._vertices_to_class()
        # Set the indices
        self._corruption_indices = [LOOKUP[side] for side in self.corruption_scheme]

    def _vertices_to_class(self):
        # use the triples factory entity to id map to surmise the entity class.
        # this assumes entity_to_id gives back a dict where the keys are of form
        # entity_class:entity_class_id and the values are their pykeen id.
        # use this pykeen id as the index and use each element as class index.
        entity_to_id = self.triples_factory.entity_to_id
        retarray = torch.empty((self.triples_factory.num_entities), dtype=torch.int64)
        for k,v in entity_to_id.items():
            name = k[:k.find(':')]
            ix = self.entity_types.index(name)
            retarray[v] = ix
        return retarray

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """Generate negative samples from the positive batch."""
        if self.num_negs_per_pos > 1:
            positive_batch = positive_batch.repeat(self.num_negs_per_pos, 1)
        self.type_map = self.type_map.to(positive_batch.device)

        # Bind number of negatives to sample
        num_negs = positive_batch.shape[0]

        # Equally corrupt all sides
        split_idx = num_negs // len(self._corruption_indices)

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()


        for index, start in zip(self._corruption_indices, range(0, num_negs, split_idx)):
            stop = min(start + split_idx, num_negs)

            sub_batch = negative_batch[start:stop]
            classes = torch.unique(self.type_map[sub_batch[:,index]])
            for cls in classes:
                cls_idxs = torch.nonzero(self.type_map[sub_batch[:,index]] == cls)
                sub_batch[cls_idxs, index] = sub_batch[cls_idxs[torch.randperm(cls_idxs.shape[0])],index]
            negative_batch[start:stop] = sub_batch

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        if self.filtered:
            negative_batch, batch_filter = self.filter_negative_triples(negative_batch=negative_batch)
        else:
            batch_filter = None
        return negative_batch, batch_filter
