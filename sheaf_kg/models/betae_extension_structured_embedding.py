# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

from typing import Any, Callable, ClassVar, Mapping, Optional

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
from pykeen.nn.representation import Embedding

from sheaf_kg.interactions.multisection_structured_embedding import BetaeExtensionInteraction
from sheaf_kg.regularizers.multisection_regularizers import OrthogonalSectionsRegularizer
from sheaf_kg.representations.parameterized_embedding import ParameterizedEmbedding
from sheaf_kg.models.model_utils import prepare_query_for_prediction

class BetaeExtensionStructuredEmbedding(ERModel):

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        C0_dimension: int = 50,
        C1_dimension: int = 20,
        num_sections: int = 1,
        scoring_fct_norm: int = 2,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        regularizer: HintOrType[Regularizer] = OrthogonalSectionsRegularizer,
        regularizer_kwargs: OptionalKwargs = None,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        restriction_initializer: Hint[Initializer] = xavier_uniform_norm_,
        restriction_parameterization: Optional[Callable] = None,
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
        relation_representation_kwargs = dict(
                    shape=(C1_dimension, C0_dimension),
                    initializer=restriction_initializer,
                    parameterization=restriction_parameterization,
                )
        if restriction_parameterization is None:
            relation_representation_kwargs.pop('parameterization')
        super().__init__(
            interaction=BetaeExtensionInteraction(
                p=scoring_fct_norm
            ),
            entity_representations_kwargs=dict(
                shape=(C0_dimension,num_sections),
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations=(Embedding if restriction_parameterization is None else ParameterizedEmbedding),
            relation_representations_kwargs=[relation_representation_kwargs, relation_representation_kwargs],
            **kwargs,
        )

        self.num_sections = num_sections

    def predict(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        full_batch: bool = True,
        ids: Optional[torch.LongTensor] = None,
        slice_size: Optional[int] = None,
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
        # unpack and reformat queries. then send to scoring function

        if target == LABEL_TAIL and isinstance(hrt_batch[0], dict):
            # assume batch is all of the same query structure
            query_structure = hrt_batch[0]['structure']
            hrt_batch = prepare_query_for_prediction(hrt_batch, query_structure, self.device)
            return self.predict_t_eval(hrt_batch, query_structure=query_structure, slice_size=slice_size)

        if target == LABEL_TAIL:
            if full_batch:
                hrt_batch = hrt_batch[:, 0:2]
            return self.predict_t(hrt_batch, slice_size=slice_size)
        
        if target == LABEL_RELATION:
            if full_batch:
                hrt_batch = hrt_batch[:, 0::2]
            return self.predict_r(hrt_batch, slice_size=slice_size)

        if target == LABEL_HEAD:
            if full_batch:
                hrt_batch = hrt_batch[:, 1:3]
            return self.predict_h(hrt_batch, slice_size=slice_size)

        raise ValueError(f"Unknown target={target}")

    def predict_t_eval(
        self,
        hr_batch: torch.LongTensor,
        slice_size: Optional[int] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward pass using right side (tail) prediction for obtaining scores of all possible tails.

        This method calculates the score for all possible tails for each (head, relation) pair.

        Additionally, the model is set to evaluation mode.

        :param hr_batch: shape: (batch_size, 2), dtype: long
            The indices of (head, relation) pairs.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Model.score_t`

        :return: shape: (batch_size, num_tails), dtype: float
            For each h-r pair, the scores for all possible tails.

        .. note::

            We only expect the right side-predictions, i.e., $(h,r,*)$ to change its
            default behavior when the model has been trained with inverse relations
            (mainly because of the behavior of the LCWA training approach). This is why
            the :func:`predict_h` has different behavior depending on
            if inverse triples were used in training, and why this function has the same
            behavior regardless of the use of inverse triples.
        """
        self.eval()  # Enforce evaluation mode
        score_fun = self.score_t_batched_eval
        if slice_size is not None:
            scores = torch.cat([score_fun(hr_batch, tails=smb, **kwargs)for smb in 
                            torch.split(torch.arange(self.num_entities), slice_size, dim=0)],
                            dim=-1)
        else:
            scores = score_fun(hr_batch, **kwargs)
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores
        
    # docstr-coverage: inherited
    def score_t_batched_eval(
        self,
        hr_batch: torch.LongTensor,
        *,
        query_structure: str = '1p',
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        tails: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        if query_structure == '1p':
            hr_batch = hr_batch.unsqueeze(dim=1)
            h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1], t=tails, mode=mode)
            # unsqueeze if necessary
            if tails is None or tails.ndimension() == 1:
                t = parallel_unsqueeze(t, dim=0)
            return repeat_if_necessary(
                scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
                representations=self.entity_representations,
                num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1],
            )
        
        if query_structure == '2p':
            edge_index = torch.LongTensor([[0,1],[1,2]]).T
            boundary_vertices = torch.LongTensor([0,2])
            interior_vertices = torch.LongTensor([1])
            source_vertices = torch.LongTensor([0])
            target_vertices = torch.LongTensor([1])

            # r[0] will be of size (nbatch, 2, c0_dim) where the second dimension records the number of 
            # relations in the path. r[1] will be the same shape but will contain the tail restriction maps.
            h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1:], t=tails, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)
            
            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)
            t = t.to(self.device)

            nbatch = restriction_maps.shape[0]
            
            scores = self.interaction.score_schur_batched(edge_index, restriction_maps, 
                                                boundary_vertices, interior_vertices, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))

        if query_structure == '3p':
            edge_index = torch.LongTensor([[0,1],[1,2],[2,3]]).T
            boundary_vertices = torch.LongTensor([0,3])
            interior_vertices = torch.LongTensor([1,2])
            source_vertices = torch.LongTensor([0])
            target_vertices = torch.LongTensor([1])

            # r[0] will be of size (nbatch, 2, c0_dim) where the second dimension records the number of 
            # relations in the path. r[1] will be the same shape but will contain the tail restriction maps.
            h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1:], t=tails, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)

            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)
            t = t.to(self.device)

            nbatch = restriction_maps.shape[0]
            
            scores = self.interaction.score_schur_batched(edge_index, restriction_maps, 
                                                boundary_vertices, interior_vertices, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))

        if query_structure == '2i':
            edge_index = torch.LongTensor([[0,2],[1,2]]).T
            boundary_vertices = torch.LongTensor([0,1,2])
            source_vertices = torch.LongTensor([0,1])
            target_vertices = torch.LongTensor([2])

            
            h, r, t = self._get_representations(h=hr_batch[:, 0, :], r=hr_batch[:, 1, :], t=tails, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)

            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)
            t = t.to(self.device)

            nbatch = restriction_maps.shape[0]
            
            scores = self.interaction.score_intersect_batched(edge_index, restriction_maps, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))

        if query_structure == '3i':
            edge_index = torch.LongTensor([[0,3],[1,3],[2,3]]).T
            boundary_vertices = torch.LongTensor([0,1,2,3])
            source_vertices = torch.LongTensor([0,1,2])
            target_vertices = torch.LongTensor([3])
            
            h, r, t = self._get_representations(h=hr_batch[:, 0, :], r=hr_batch[:, 1, :], t=tails, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)

            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)
            t = t.to(self.device)

            nbatch = restriction_maps.shape[0]

            scores = self.interaction.score_intersect_batched(edge_index, restriction_maps, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))
        
        if query_structure == 'pi':
            edge_index = torch.LongTensor([[0,2],[2,3],[1,3]]).T
            boundary_vertices = torch.LongTensor([0,1,3])
            interior_vertices = torch.LongTensor([2])
            source_vertices = torch.LongTensor([0,1])
            target_vertices = torch.LongTensor([2])
            
            dummy_idx = torch.LongTensor([0])
            _, _, t = self._get_representations(h=dummy_idx, r=dummy_idx, t=tails, mode=mode)
            t = t.to(self.device)
            
            sources = torch.cat([b['sources'].unsqueeze(0) for b in hr_batch], axis=0)
            relations = torch.cat([b['relations'].unsqueeze(0) for b in hr_batch], axis=0)
            h, r, _ = self._get_representations(h=sources, r=relations, t=dummy_idx, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)
            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)

            nbatch = restriction_maps.shape[0]
            
            scores = self.interaction.score_schur_batched(edge_index, restriction_maps, 
                                                boundary_vertices, interior_vertices, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))
        

        if query_structure == 'ip':
            edge_index = torch.LongTensor([[0,2],[1,2],[2,3]]).T
            boundary_vertices = torch.LongTensor([0,1,3])
            interior_vertices = torch.LongTensor([2])
            source_vertices = torch.LongTensor([0,1])
            target_vertices = torch.LongTensor([2])

            dummy_idx = torch.LongTensor([0])
            _, _, t = self._get_representations(h=dummy_idx, r=dummy_idx, t=tails, mode=mode)
            t = t.to(self.device)
            
            sources = torch.cat([b['sources'].unsqueeze(0) for b in hr_batch], axis=0)
            relations = torch.cat([b['relations'].unsqueeze(0) for b in hr_batch], axis=0)
            h, r, _ = self._get_representations(h=sources, r=relations, t=dummy_idx, mode=mode)
            restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)
            # move everything to proper device
            restriction_maps = restriction_maps.to(self.device)
            h = h.to(self.device)

            nbatch = restriction_maps.shape[0]
            
            scores = self.interaction.score_schur_batched(edge_index, restriction_maps, 
                                                boundary_vertices, interior_vertices, 
                                                source_vertices, target_vertices, 
                                                h.reshape(nbatch, -1, self.num_sections), t, t.shape[-2])
            return -torch.sum(scores, dim=(-1))

            