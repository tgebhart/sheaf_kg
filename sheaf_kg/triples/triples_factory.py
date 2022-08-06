from typing import Union, Optional, Any
import warnings

import numpy as np
from torch.utils.data import Dataset
from pykeen.typing import MappedTriples, Mapping, EntityMapping, RelationMapping
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.instances import SubGraphSLCWAInstances, BatchedSLCWAInstances

class ComplexTriplesFactory(TriplesFactory):
    """Create instances from ID-based triples."""

    def __init__(
        self,
        mapped_triples: MappedTriples,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        create_inverse_triples: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs
    ):
        """
        Create the triples factory.

        :param mapped_triples: shape: (n, 3)
            A three-column matrix where each row are the head identifier, relation identifier, then tail identifier.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param metadata:
            Arbitrary metadata to go with the graph

        :raises TypeError:
            if the mapped_triples are of non-integer dtype
        :raises ValueError:
            if the mapped_triples are of invalid shape
        """
        super().__init__(
            mapped_triples=mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=create_inverse_triples,
            metadata=metadata,
            **kwargs
        )

    def create_slcwa_instances(self, *, sampler: Optional[str] = None, **kwargs) -> Dataset:
        """Create sLCWA instances for this factory's triples."""
        print(f'GOT SAMPLER {sampler}')
        if sampler == 'complex':
            raise ValueError('not yet implemented')
        elif sampler == 'schlichtkrull':
            cls = SubGraphSLCWAInstances
        else:
            cls = BatchedSLCWAInstances
        print(f'SET SAMPLER {cls}')
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn("Training instances are always shuffled.", DeprecationWarning)
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        return cls(
            mapped_triples=self._add_inverse_triples_if_necessary(mapped_triples=self.mapped_triples),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **kwargs,
        )
