import numpy as np
import pykeen
import torch

from pykeen.pipeline import pipeline
from typing import Optional
from pykeen.models import StructuredEmbedding
from pykeen.models.base import _OldAbstractModel
from pykeen.nn import Embedding
from pykeen.losses import Loss
from pykeen.nn.init import xavier_uniform_
from pykeen.regularizers import Regularizer
from pykeen.triples import TriplesFactory
from pykeen.typing import DeviceHint
from pykeen.utils import compose

from torch.nn import functional
from torch.nn.parameter import Parameter
from torch import nn


class SheafE_Multisection(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        num_sections: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.symmetric = symmetric
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm

        esize = (triples_factory.num_entities, num_sections, embedding_dim)
        self.ent_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(esize, device=preferred_device, dtype=torch.float32)),requires_grad=True)

        tsize = (triples_factory.num_relations, edge_stalk_dim, embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, dtype=torch.float32)),requires_grad=True)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Multisection_Symmetric_{}embdim_{}esdim_{}nsec_{}norm'
        else:
            savestruct = 'SheafE_Multisection_{}embdim_{}esdim_{}sec_{}norm'
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):  # noqa: D102
        self.ent_embeddings = Parameter(nn.init.xavier_uniform_(self.ent_embeddings))
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(self.left_embeddings))
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = nn.init.xavier_uniform_(self.right_embeddings)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1])
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1])
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, self.num_sections)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t
        scores = -torch.norm(proj_h - proj_t, dim=(1,2), p=self.scoring_fct_norm)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hr_batch[:, 1])
        rel_t = torch.index_select(self.right_embeddings, 0, hr_batch[:, 1])
        rel_t = rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        t_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            proj_h = rel_h @ h
            proj_t = rel_t @ t_all

        scores = -torch.norm(proj_h[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, rt_batch[:, 0])
        rel_h = rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        rel_t = torch.index_select(self.right_embeddings, 0, rt_batch[:, 0])
        t = torch.index_select(self.ent_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            proj_h = rel_h @ h_all
            proj_t = rel_t @ t

        scores = -torch.norm(proj_h[:, :, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores


class SheafE_Diag(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        num_sections: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.symmetric = symmetric
        self.embedding_dim = embedding_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm

        esize = (triples_factory.num_entities, num_sections, embedding_dim)
        self.ent_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(esize, device=preferred_device, dtype=torch.float32)),requires_grad=True)

        tsize = (triples_factory.num_relations, embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, dtype=torch.float32)),requires_grad=True)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Diag_Symmetric_{}embdim_{}nsec_{}norm'
        else:
            savestruct = 'SheafE_Diag_{}embdim_{}sec_{}norm'
        return savestruct.format(self.embedding_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):  # noqa: D102
        self.ent_embeddings = nn.init.xavier_uniform_(self.ent_embeddings)
        self.left_embeddings = nn.init.xavier_uniform_(self.left_embeddings)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = nn.init.xavier_uniform_(self.right_embeddings)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1]).view(-1)
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1]).view(-1)
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.num_sections)

        # Project entities
        proj_h = (torch.diagflat(rel_h) @ h).view(-1,self.embedding_dim,self.num_sections)
        proj_t = (torch.diagflat(rel_t) @ t).view(-1,self.embedding_dim,self.num_sections)

        scores = -torch.norm(proj_h - proj_t, dim=(1,2), p=self.scoring_fct_norm)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hr_batch[:, 0]).view(-1, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hr_batch[:, 1]).view(-1)
        rel_t = torch.index_select(self.right_embeddings, 0, hr_batch[:, 1]).view(-1, self.embedding_dim)
        t_all = self.ent_embeddings.view(-1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            proj_h = (torch.diagflat(rel_h) @ h).view(-1,self.embedding_dim,self.num_sections)
            diagt = torch.diag_embed(rel_t)
            diagt = diagt.view(diagt.shape[0],1,diagt.shape[1],diagt.shape[2])
            proj_t = torch.matmul(diagt, t_all)

        scores = -torch.norm(proj_h[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h_all = self.ent_embeddings.view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, rt_batch[:, 0]).view(-1, self.embedding_dim)
        rel_t = torch.index_select(self.right_embeddings, 0, rt_batch[:, 0]).view(-1)
        t = torch.index_select(self.ent_embeddings, 0, rt_batch[:, 1]).view(-1, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            diagh = torch.diag_embed(rel_h)
            diagh = diagh.view(diagh.shape[0],1,diagh.shape[1],diagh.shape[2])
            proj_h = torch.matmul(diagh, h_all)
            proj_t = (torch.diagflat(rel_t) @ t).view(-1,self.embedding_dim,self.num_sections)

        scores = -torch.norm(proj_h[:, :, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores
