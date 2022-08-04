# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

from typing import Any, Callable, ClassVar, Mapping, Optional, Literal

from class_resolver import Hint, HintOrType, OptionalKwargs
import torch
from torch.nn import functional

from pykeen.models.nbase import ERModel, repeat_if_necessary
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_
from pykeen.nn.modules import parallel_unsqueeze
from pykeen.typing import Constrainer, Initializer
from pykeen.regularizers import Regularizer
from pykeen.typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, \
                        MappedTriples, Target, InductiveMode

from sheaf_kg.interactions.multisection_structured_embedding import (
    ExtensionInteraction,
)
from sheaf_kg.regularizers.multisection_regularizers import (
    OrthogonalSectionsRegularizer,
)
from sheaf_kg.representations.parameterized_embedding import ParameterizedEmbedding

class ExtensionStructuredEmbedding(ERModel):

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        C0_dimension: int = 50,
        C1_dimension=20,
        num_sections: int = 3,
        scoring_fct_norm: int = 2,
        training_mask_pct : float = 0,
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
            interaction=ExtensionInteraction(
                p=scoring_fct_norm,
                training_mask_pct=training_mask_pct,
            ),
            entity_representations_kwargs=dict(
                shape=(C0_dimension,num_sections),
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

    def predict(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        full_batch: bool = True,
        ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Predict scores for the given target.

        :param hrt_batch: shape: (batch_size, 3) or (batch_size, 2)
            the full batch, or the relevant part of it
        :param target:
            the target to predict
        :param full_batch:
            whether `hrt_batch` is the full batch, or only the "input" part of the target prediction method
        :param ids:
            restrict prediction to only those ids
        :param kwargs:
            additional keyword-based parameters passed to the specific target prediction method.

        :raises ValueError:
            if the target is invalid

        :return: shape: (batch_size, num)
            the scores
        """
        if target == LABEL_TAIL:
            if full_batch:
                hrt_batch = hrt_batch
            return self.predict_t(hrt_batch, **kwargs)

        if target == LABEL_RELATION:
            if full_batch:
                hrt_batch = hrt_batch[:, 0::2]
            return self.predict_r(hrt_batch, **kwargs, relations=ids)

        if target == LABEL_HEAD:
            if full_batch:
                hrt_batch = hrt_batch
            return self.predict_h(hrt_batch, **kwargs)

        raise ValueError(f"Unknown target={target}")

# docstr-coverage: inherited
    def score_h(
        self,
        rt_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        heads: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        rt_batch = rt_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=rt_batch[..., 0], r=rt_batch[..., 1], t=rt_batch[..., 2], mode=mode)
        h_score, _, _ = self._get_representations(h=heads, r=rt_batch[..., 1], t=rt_batch[..., 2], mode=mode)
        if heads is None or heads.ndimension() == 1:
            h_score = parallel_unsqueeze(h_score, dim=0)
        # unsqueeze if necessary
        return repeat_if_necessary(
            scores=self.interaction.score_h(h=h, r=r, t=t, to_score=h_score, slice_size=slice_size, slice_dim=1),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )