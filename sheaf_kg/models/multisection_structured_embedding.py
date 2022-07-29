# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

from typing import Any, Callable, ClassVar, Mapping, Optional

from class_resolver import Hint, HintOrType, OptionalKwargs
from torch import nn
from torch.nn import functional

from pykeen.models.nbase import ERModel
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_
from pykeen.typing import Constrainer, Initializer
from pykeen.regularizers import Regularizer

from sheaf_kg.interactions.multisection_structured_embedding import (
    MultisectionStructuredEmbeddingInteraction,
)
from sheaf_kg.regularizers.multisection_regularizers import (
    OrthogonalSectionsRegularizer,
)
from sheaf_kg.representations.parameterized_embedding import ParameterizedEmbedding

__all__ = [
    "MultisectionSE",
]


class MultisectionStructuredEmbedding(ERModel):

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        C0_dimension: int = 50,
        num_sections: int = 3,
        C1_dimension=20,
        scoring_fct_norm: int = 2,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        regularizer: HintOrType[Regularizer] = OrthogonalSectionsRegularizer,
        regularizer_kwargs: OptionalKwargs = None,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_parametrization: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for SE.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`torch.nn.functional.normalize`
        :param entity_constrainer_kwargs: Keyword arguments to be used when calling the entity constrainer
        :param relation_initializer: Relation initializer function. Defaults to
            :func:`pykeen.nn.init.xavier_uniform_norm_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityEmbeddingModel`
        """
        super().__init__(
            interaction=MultisectionStructuredEmbeddingInteraction(
                p=scoring_fct_norm,
                # power_norm=True,
            ),
            entity_representations_kwargs=dict(
                shape=(C0_dimension, num_sections),
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations=ParameterizedEmbedding,
            relation_representations_kwargs=[
                dict(
                    shape=(C1_dimension, C0_dimension),
                    initializer=relation_initializer,
                    parametrization=relation_parametrization,
                ),
                dict(
                    shape=(C1_dimension, C0_dimension),
                    initializer=relation_initializer,
                    parametrization=relation_parametrization,
                ),
            ],
            **kwargs,
        )
