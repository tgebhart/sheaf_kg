from functools import partial
from typing import Optional

import numpy as np
import pykeen
import torch

from pykeen.pipeline import pipeline
from pykeen.models import StructuredEmbedding
from pykeen.models.base import _OldAbstractModel
from pykeen.nn import Embedding
from pykeen.losses import Loss
from pykeen.nn.init import xavier_uniform_
from pykeen.regularizers import Regularizer
from pykeen.triples import TriplesFactory
from pykeen.typing import DeviceHint, Constrainer
from pykeen.utils import compose

from torch.nn import functional
from torch.nn.parameter import Parameter
from torch import nn
import torch.distributions as distributions

import sheaf_kg.batch_harmonic_extension as harmonic_extension
from sheaf_kg.complex_functions import L_p_multisection, L_i_multisection, L_ip_multisection, L_pi_multisection
from sheaf_kg.complex_functions import L_p1_multisection, L_p_cvx, L_ip_cvx, L_pi_cvx, L_p_translational_cvx, L_ip_translational_cvx, L_pi_translational_cvx
from sheaf_kg.complex_functions import L_p1_translational, L_p_translational, L_i_translational, L_ip_translational, L_pi_translational
from sheaf_kg.complex_functions import cvxpy_problem, linear_chain, pi_chain, ip_chain

class SheafE_Multisection(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
        num_sections: int = 1,
        complex_solver: str = 'schur',
        lbda: float = 0.5,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_constrainer: Optional[Constrainer] = functional.normalize
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.symmetric = bool(symmetric)
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.lbda = lbda
        self.device = preferred_device
        self.entity_constrainer = entity_constrainer
        self.complex_solver = complex_solver

        if self.complex_solver == 'schur':
            self.query_name_fn_dict = { '1p':L_p1_multisection,
                                        '2p':L_p_multisection,
                                        '3p':L_p_multisection,
                                        '2i':L_i_multisection,
                                        '3i':L_i_multisection,
                                        'ip':L_ip_multisection,
                                        'pi':L_pi_multisection }
        if self.complex_solver == 'cvx':
            lp2_layer = cvxpy_problem(linear_chain(2), self.embedding_dim, self.edge_stalk_dim, [0])
            lp3_layer = cvxpy_problem(linear_chain(3), self.embedding_dim, self.edge_stalk_dim, [0])
            lip_layer = cvxpy_problem(ip_chain(), self.embedding_dim, self.edge_stalk_dim, [0,1])
            lpi_layer = cvxpy_problem(pi_chain(), self.embedding_dim, self.edge_stalk_dim, [0,1])
            self.query_name_fn_dict = { '1p':L_p1_multisection,
                                        '2p':partial(L_p_cvx, layer=lp2_layer),
                                        '3p':partial(L_p_cvx, layer=lp3_layer),
                                        '2i':L_i_multisection,
                                        '3i':L_i_multisection,
                                        'ip':partial(L_ip_cvx, layer=lip_layer),
                                        'pi':partial(L_pi_cvx, layer=lpi_layer) }

        self.initialize_entities()
        self.initialize_relations()

    def initialize_entities(self):
        esize = (self.num_entities, self.embedding_dim, self.num_sections)
        if self.orthogonal and self.num_sections > 1:
            # is there a faster way to do this? looping over num entities is expensive
            orths = torch.empty(esize, device=self.device, dtype=torch.float32)
            for i in range(esize[0]):
                orths[i,:,:] = nn.init.orthogonal_(orths[i,:,:])
            self.ent_embeddings = Parameter(orths, requires_grad=True)
            self.I = torch.eye(self.num_sections, device=self.device)
        else:
            self.ent_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(esize, device=self.device, dtype=torch.float32)),requires_grad=True)

    def initialize_relations(self):
        tsize = (self.num_relations, self.edge_stalk_dim, self.embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)),requires_grad=True)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Multisection_Symmetric_{}embdim_{}esdim_{}sec_{}norm_{}lbda'
        else:
            savestruct = 'SheafE_Multisection_{}embdim_{}esdim_{}sec_{}norm_{}lbda'
        if self.orthogonal:
            savestruct += '_{}orthogonal'.format(self.alpha_orthogonal)
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm, self.lbda)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

    def post_parameter_update(self):
        super().post_parameter_update()
        if self.entity_constrainer is not None:
            self.ent_embeddings.data = self.entity_constrainer(self.ent_embeddings.data, dim=1)

    def forward_costs(self, query_name, entities, relations, targets, invs=None):
        return self.query_name_fn_dict[query_name](self, entities, relations, targets, invs=invs)

    def score_query(self, query_name, entities, relations, targets, invs=None):
        Q = self.forward_costs(query_name, entities, relations, targets, invs=invs)
        score = -torch.linalg.norm(Q, ord=self.scoring_fct_norm, dim=(-1))
        if self.orthogonal and self.num_sections > 1:
            h = torch.index_select(self.ent_embeddings, 0, entities.flatten())
            t = torch.index_select(self.ent_embeddings, 0, targets)
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            orth_scores = self.alpha_orthogonal * torch.sum(torch.norm(ents.permute(0,2,1)@ents - I, dim=(-2,-1), p=2))
            return score - orth_scores
        return score

    def project_hrt(self, hrt_batch: torch.LongTensor):
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1])
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1])
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, self.num_sections)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t
        return proj_h, proj_t

    def score_hrt_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor) -> torch.FloatTensor:
        scores = - ((1-self.lbda)*torch.norm(proj_h - proj_t, dim=(1,2), p=self.scoring_fct_norm)**2 + self.lbda*torch.sum(torch.norm(proj_h - proj_t, dim=(1,2), p=self.scoring_fct_norm)**2, dim=0))
        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_hrt(hrt_batch)
        scores = self.score_hrt_projections(proj_h, proj_t)
        if self.orthogonal and self.num_sections > 1:
            h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
            t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, self.num_sections)
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            orth_scores = self.alpha_orthogonal * torch.sum(torch.norm(ents.permute(0,2,1)@ents - I, dim=(-2,-1), p=2))
            return scores - orth_scores
        return scores

    def project_t(self, hr_batch: torch.LongTensor, slice_size: int = None):
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
        return proj_h, proj_t

    def score_hr_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor) -> torch.FloatTensor:
        scores = -torch.norm(proj_h[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)**2
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_t(hr_batch, slice_size)
        return self.score_hr_projections(proj_h, proj_t)

    def project_h(self, rt_batch: torch.LongTensor, slice_size: int = None):
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
        return proj_h, proj_t

    def score_rt_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor) -> torch.FloatTensor:
        scores = -torch.norm(proj_h[:, :, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)**2
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_h(rt_batch, slice_size)
        return self.score_rt_projections(proj_h, proj_t)

class SheafE_Diag(SheafE_Multisection):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
        num_sections: int = 1,
        lbda: float = 0.5,
        complex_solver: str = 'schur',
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_constrainer: Optional[Constrainer] = functional.normalize
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            edge_stalk_dim=embedding_dim,
            scoring_fct_norm=scoring_fct_norm,
            symmetric=symmetric,
            orthogonal=orthogonal,
            alpha_orthogonal=alpha_orthogonal,
            num_sections=num_sections,
            complex_solver=complex_solver,
            lbda=lbda,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

    def initialize_relations(self):
        tsize = (self.num_relations, self.embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)),requires_grad=True)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Diag_Symmetric_{}embdim_{}sec_{}norm_{}lbda'
        else:
            savestruct = 'SheafE_Diag_{}embdim_{}sec_{}norm_{}lbda'
        return savestruct.format(self.embedding_dim, self.num_sections, self.scoring_fct_norm, self.lbda)

    def project_hrt(self, hrt_batch: torch.LongTensor):
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1]).view(-1)
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1]).view(-1)
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.num_sections)

        # Project entities
        proj_h = (torch.diagflat(rel_h) @ h).view(-1,self.embedding_dim,self.num_sections)
        proj_t = (torch.diagflat(rel_t) @ t).view(-1,self.embedding_dim,self.num_sections)
        return proj_h, proj_t

    def project_t(self, hr_batch: torch.LongTensor, slice_size: int = None):
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
        return proj_h, proj_t

    def project_h(self, rt_batch: torch.LongTensor, slice_size: int = None):
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
        return proj_h, proj_t

class SheafE_Bilinear(SheafE_Multisection):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = True, # by definition
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
        lbda: float = 0.5,
        num_sections: int = 1,
        complex_solver: str = 'schur',
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_constrainer: Optional[Constrainer] = functional.normalize
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            edge_stalk_dim=edge_stalk_dim,
            scoring_fct_norm=scoring_fct_norm,
            symmetric=symmetric,
            orthogonal=orthogonal,
            alpha_orthogonal=alpha_orthogonal,
            num_sections=num_sections,
            complex_solver=complex_solver,
            lbda=lbda,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        if self.complex_solver == 'schur':
            self.query_name_fn_dict = { '1p':L_p1_translational,
                                        '2p':L_p_translational,
                                        '3p':L_p_translational,
                                        '2i':L_i_translational,
                                        '3i':L_i_translational,
                                        'ip':L_ip_translational,
                                        'pi':L_pi_translational }
        if self.complex_solver == 'cvx':
            lp2_layer = cvxpy_problem(linear_chain(2), self.embedding_dim, self.edge_stalk_dim, [0], edge_cochains=True)
            lp3_layer = cvxpy_problem(linear_chain(3), self.embedding_dim, self.edge_stalk_dim, [0], edge_cochains=True)
            lip_layer = cvxpy_problem(ip_chain(), self.embedding_dim, self.edge_stalk_dim, [0,1], edge_cochains=True)
            lpi_layer = cvxpy_problem(pi_chain(), self.embedding_dim, self.edge_stalk_dim, [0,1], edge_cochains=True)
            self.query_name_fn_dict = { '1p':L_p1_translational,
                                        '2p':partial(L_p_translational_cvx, layer=lp2_layer),
                                        '3p':partial(L_p_translational_cvx, layer=lp3_layer),
                                        '2i':L_i_translational,
                                        '3i':L_i_translational,
                                        'ip':partial(L_ip_translational_cvx, layer=lip_layer),
                                        'pi':partial(L_pi_translational_cvx, layer=lpi_layer) }

        self.initialize_edge_cochains()

    def initialize_edge_cochains(self):
        tsize = (self.num_relations, self.edge_stalk_dim, self.num_sections)
        self.edge_cochains = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)

    def get_model_savename(self):
        savestruct = 'SheafE_Bilinear_{}embdim_{}esdim_{}sec_{}norm_{}lbda'
        if self.orthogonal:
            savestruct += '_{}orthogonal'.format(self.alpha_orthogonal)
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm, self.lbda)

    def _reset_parameters_(self):
        self.initialize_entities()
        self.initialize_relations()
        self.initialize_edge_cochains()

    def post_parameter_update(self):
        super().post_parameter_update()
        if self.entity_constrainer is not None:
            self.ent_embeddings.data = self.entity_constrainer(self.ent_embeddings.data, dim=1)
            self.edge_cochains.data = self.entity_constrainer(self.edge_cochains.data, dim=1)

    def score_hrt_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        scores = -((1-self.lbda)*torch.norm(proj_h + c - proj_t, dim=(1,2), p=self.scoring_fct_norm)**2 +
                    self.lbda*torch.sum(torch.norm(proj_h + c - proj_t, dim=(1,2), p=self.scoring_fct_norm)**2, dim=0))
        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_hrt(hrt_batch)
        c = torch.index_select(self.edge_cochains, 0, hrt_batch[:,1]).view(-1, self.edge_stalk_dim, self.num_sections)
        scores = self.score_hrt_projections(proj_h, proj_t, c)
        if self.orthogonal and self.num_sections > 1:
            h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
            t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, self.num_sections)
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            orth_scores = self.alpha_orthogonal * torch.sum(torch.norm(ents.permute(0,2,1)@ents - I, dim=(-2,-1), p=2))
            return scores - orth_scores
        return scores

    def score_hr_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        scores = -torch.norm(proj_h[:, None, :, :] + c[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)**2
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_t(hr_batch, slice_size)
        c = torch.index_select(self.edge_cochains, 0, hr_batch[:,1]).view(-1, self.edge_stalk_dim, self.num_sections)
        return self.score_hr_projections(proj_h, proj_t, c)

    def score_rt_projections(self, proj_h: torch.FloatTensor, proj_t: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        scores = -torch.norm(proj_h[:, :, :, :] + c[:, None, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)**2
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        proj_h, proj_t = self.project_h(rt_batch, slice_size)
        c = torch.index_select(self.edge_cochains, 0, rt_batch[:,0]).view(-1, self.edge_stalk_dim, self.num_sections)
        return self.score_rt_projections(proj_h, proj_t, c)

class SheafE_Translational(SheafE_Bilinear):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = True, # by definition
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
        lbda: float = 0.5,
        num_sections: int = 1,
        rel_identity: bool = True,
        complex_solver: str = 'schur',
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_constrainer: Optional[Constrainer] = functional.normalize
    ) -> None:

        self.symmetric = True
        self.rel_identity = rel_identity

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            edge_stalk_dim=embedding_dim,
            scoring_fct_norm=scoring_fct_norm,
            symmetric=True,
            orthogonal=orthogonal,
            alpha_orthogonal=alpha_orthogonal,
            num_sections=num_sections,
            lbda=lbda,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

    def initialize_relations(self):
        if self.rel_identity:
            id = torch.eye(self.edge_stalk_dim, self.embedding_dim, device=self.device, dtype=torch.float32, requires_grad=False)
            id = id.reshape((1, self.edge_stalk_dim, self.embedding_dim))
            batch_id = id.repeat(self.num_relations, 1, 1)
            self.left_embeddings = batch_id
            self.right_embeddings = batch_id
        else:
            diag = nn.init.xavier_uniform_(torch.empty((self.num_relations, self.embedding_dim), dtype=torch.float32, requires_grad=False))
            mat = torch.diag_embed(diag).to(self.device)
            self.left_embeddings = mat
            self.right_embeddings = mat

    def get_model_savename(self):
        if self.rel_identity:
            savestruct = 'SheafE_Translational_{}embdim_{}esdim_{}sec_{}norm_{}lbda'
        else:
            savestruct = 'SheafE_Translational_Rand_Diag_{}embdim_{}esdim_{}sec_{}norm_{}lbda'
        if self.orthogonal:
            savestruct += '_{}orthogonal'.format(self.alpha_orthogonal)
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm, self.lbda)

class SheafE_Distributional_Normal(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 32,
        edge_stalk_dim: int = 32,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        entity_constrainer: Optional[Constrainer] = None
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed
        )

        self.symmetric = bool(symmetric)
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.device = preferred_device
        self.entity_constrainer = entity_constrainer

        self.initialize_entities()
        self.initialize_relations()

    def initialize_entities(self):
        esize = (self.num_entities, self.embedding_dim)
        self.mu_embeddings = Parameter((torch.zeros(esize, device=self.device, dtype=torch.float32)), requires_grad=True)
        self.sigma_embeddings = Parameter((torch.ones(esize, device=self.device, dtype=torch.float32)), requires_grad=True)

    def initialize_relations(self):
        # for now, initialize restrictions for sigma to be identity to keep from negative
        tsize = (self.num_relations, self.edge_stalk_dim, self.embedding_dim)
        I = torch.eye(tsize[1], tsize[2], device=self.device).reshape((1, tsize[1], tsize[2]))
        I = I.repeat(tsize[0], 1, 1)
        self.mu_left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
        # self.sigma_left_embeddings = Parameter(nn.init.uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
        self.sigma_left_embeddings = Parameter(I, requires_grad=False)
        if self.symmetric:
            self.mu_right_embeddings = self.mu_left_embeddings
            self.sigma_right_embeddings = self.sigma_left_embeddings
        else:
            self.mu_right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
            # self.sigma_right_embeddings = Parameter(nn.init.uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
            self.sigma_right_embeddings = Parameter(I, requires_grad=False)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Distributional_Normal_Symmetric_{}embdim_{}esdim'
        else:
            savestruct = 'SheafE_Distributional_Normal_{}embdim_{}esdim'
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

    def post_parameter_update(self):
        super().post_parameter_update()
        if self.entity_constrainer is not None:
            self.mu_embeddings.data = self.entity_constrainer(self.mu_embeddings.data, dim=1)
            self.sigma_embeddings.data = torch.clamp(self.entity_constrainer(self.sigma_embeddings.data, dim=1), min=0)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        mu_h = torch.index_select(self.mu_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        mu_rel_h = torch.index_select(self.mu_left_embeddings, 0, hrt_batch[:, 1])
        mu_rel_t = torch.index_select(self.mu_right_embeddings, 0, hrt_batch[:, 1])
        mu_t = torch.index_select(self.mu_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        sigma_h = torch.index_select(self.sigma_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        sigma_rel_h = torch.index_select(self.sigma_left_embeddings, 0, hrt_batch[:, 1])
        sigma_rel_t = torch.index_select(self.sigma_right_embeddings, 0, hrt_batch[:, 1])
        sigma_t = torch.index_select(self.sigma_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Project entities
        mu_proj_h = (mu_rel_h @ mu_h).squeeze(-1)
        mu_proj_t = (mu_rel_t @ mu_t).squeeze(-1)

        sigma_proj_h = (sigma_rel_h @ sigma_h).squeeze(-1)
        sigma_proj_t = (sigma_rel_t @ sigma_t).squeeze(-1)

        h_dist = distributions.Normal(mu_proj_h, sigma_proj_h)
        t_dist = distributions.Normal(mu_proj_t, sigma_proj_t)
        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        mu_h = torch.index_select(self.mu_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        mu_rel_h = torch.index_select(self.mu_left_embeddings, 0, hr_batch[:, 1])
        mu_rel_t = torch.index_select(self.mu_right_embeddings, 0, hr_batch[:, 1])
        mu_rel_t = mu_rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        mu_t_all = self.mu_embeddings.view(1, -1, self.embedding_dim, 1)

        sigma_h = torch.index_select(self.sigma_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        sigma_rel_h = torch.index_select(self.sigma_left_embeddings, 0, hr_batch[:, 1])
        sigma_rel_t = torch.index_select(self.sigma_right_embeddings, 0, hr_batch[:, 1])
        sigma_rel_t = sigma_rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        sigma_t_all = self.sigma_embeddings.view(1, -1, self.embedding_dim, 1)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            mu_proj_h = (mu_rel_h @ mu_h).squeeze(-1)
            mu_proj_t = (mu_rel_t @ mu_t_all).squeeze(-1)

            sigma_proj_h = (sigma_rel_h @ sigma_h).squeeze(-1)
            sigma_proj_t = (sigma_rel_t @ sigma_t_all).squeeze(-1)

        h_dist = distributions.Normal(mu_proj_h.unsqueeze(1), sigma_proj_h.unsqueeze(1))
        t_dist = distributions.Normal(mu_proj_t, sigma_proj_t)

        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)

        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        mu_h_all = self.mu_embeddings.view(1, -1, self.embedding_dim, 1)
        mu_rel_h = torch.index_select(self.mu_left_embeddings, 0, rt_batch[:, 0])
        mu_rel_h = mu_rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        mu_rel_t = torch.index_select(self.mu_right_embeddings, 0, rt_batch[:, 0])
        mu_t = torch.index_select(self.mu_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        sigma_h_all = self.sigma_embeddings.view(1, -1, self.embedding_dim, 1)
        sigma_rel_h = torch.index_select(self.sigma_left_embeddings, 0, rt_batch[:, 0])
        sigma_rel_h = sigma_rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        sigma_rel_t = torch.index_select(self.sigma_right_embeddings, 0, rt_batch[:, 0])
        sigma_t = torch.index_select(self.sigma_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            mu_proj_h = (mu_rel_h @ mu_h_all).squeeze(-1)
            mu_proj_t = (mu_rel_t @ mu_t).squeeze(-1)

            sigma_proj_h = (sigma_rel_h @ sigma_h_all).squeeze(-1)
            sigma_proj_t = (sigma_rel_t @ sigma_t).squeeze(-1)

        h_dist = distributions.Normal(mu_proj_h.unsqueeze(1), sigma_proj_h.unsqueeze(1))
        t_dist = distributions.Normal(mu_proj_t, sigma_proj_t)

        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)
        return scores

class SheafE_Distributional_Beta(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 32,
        edge_stalk_dim: int = 32,
        scoring_fct_norm: int = 2,
        symmetric: bool = False,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        entity_constrainer: Optional[Constrainer] = None
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed
        )

        self.symmetric = bool(symmetric)
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.device = preferred_device
        self.entity_constrainer = entity_constrainer

        self.initialize_entities()
        self.initialize_relations()

    def initialize_entities(self):
        esize = (self.num_entities, self.embedding_dim)
        self.alpha_embeddings = Parameter(nn.init.uniform_(torch.empty(esize, device=self.device, dtype=torch.float32), a=0.05, b=1.0), requires_grad=True)
        self.beta_embeddings = Parameter(nn.init.uniform_(torch.empty(esize, device=self.device, dtype=torch.float32), a=0.05, b=1.0), requires_grad=True)

    def initialize_relations(self):
        # for now, initialize restrictions for beta to be identity to keep from negative
        tsize = (self.num_relations, self.edge_stalk_dim, self.embedding_dim)
        I = torch.eye(tsize[1], tsize[2], device=self.device).reshape((1, tsize[1], tsize[2]))
        I = I.repeat(tsize[0], 1, 1)
        self.alpha_left_embeddings = Parameter(I, requires_grad=False)
        # self.beta_left_embeddings = Parameter(nn.init.uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
        self.beta_left_embeddings = Parameter(I, requires_grad=False)
        if self.symmetric:
            self.alpha_right_embeddings = self.alpha_left_embeddings
            self.beta_right_embeddings = self.beta_left_embeddings
        else:
            self.alpha_right_embeddings = Parameter(I, requires_grad=False)
            # self.beta_right_embeddings = Parameter(nn.init.uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)
            self.beta_right_embeddings = Parameter(I, requires_grad=False)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Distributional_Beta_Symmetric_{}embdim_{}esdim'
        else:
            savestruct = 'SheafE_Distributional_Beta_{}embdim_{}esdim'
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

    def post_parameter_update(self):
        super().post_parameter_update()
        if self.entity_constrainer is not None:
            self.alpha_embeddings.data = torch.clamp(self.entity_constrainer(self.alpha_embeddings.data, dim=1), min=0.05)
            self.beta_embeddings.data = torch.clamp(self.entity_constrainer(self.beta_embeddings.data, dim=1), min=0.05)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        alpha_h = torch.index_select(self.alpha_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        alpha_rel_h = torch.index_select(self.alpha_left_embeddings, 0, hrt_batch[:, 1])
        alpha_rel_t = torch.index_select(self.alpha_right_embeddings, 0, hrt_batch[:, 1])
        alpha_t = torch.index_select(self.alpha_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        beta_h = torch.index_select(self.beta_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        beta_rel_h = torch.index_select(self.beta_left_embeddings, 0, hrt_batch[:, 1])
        beta_rel_t = torch.index_select(self.beta_right_embeddings, 0, hrt_batch[:, 1])
        beta_t = torch.index_select(self.beta_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Project entities
        alpha_proj_h = (alpha_rel_h @ alpha_h).squeeze(-1)
        alpha_proj_t = (alpha_rel_t @ alpha_t).squeeze(-1)

        beta_proj_h = (beta_rel_h @ beta_h).squeeze(-1)
        beta_proj_t = (beta_rel_t @ beta_t).squeeze(-1)

        h_dist = distributions.beta.Beta(alpha_proj_h, beta_proj_h)
        t_dist = distributions.beta.Beta(alpha_proj_t, beta_proj_t)
        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        alpha_h = torch.index_select(self.alpha_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        alpha_rel_h = torch.index_select(self.alpha_left_embeddings, 0, hr_batch[:, 1])
        alpha_rel_t = torch.index_select(self.alpha_right_embeddings, 0, hr_batch[:, 1])
        alpha_rel_t = alpha_rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        alpha_t_all = self.alpha_embeddings.view(1, -1, self.embedding_dim, 1)

        beta_h = torch.index_select(self.beta_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        beta_rel_h = torch.index_select(self.beta_left_embeddings, 0, hr_batch[:, 1])
        beta_rel_t = torch.index_select(self.beta_right_embeddings, 0, hr_batch[:, 1])
        beta_rel_t = beta_rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        beta_t_all = self.beta_embeddings.view(1, -1, self.embedding_dim, 1)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            alpha_proj_h = (alpha_rel_h @ alpha_h).squeeze(-1)
            alpha_proj_t = (alpha_rel_t @ alpha_t_all).squeeze(-1)

            beta_proj_h = (beta_rel_h @ beta_h).squeeze(-1)
            beta_proj_t = (beta_rel_t @ beta_t_all).squeeze(-1)

        h_dist = distributions.beta.Beta(alpha_proj_h.unsqueeze(1), beta_proj_h.unsqueeze(1))
        t_dist = distributions.beta.Beta(alpha_proj_t, beta_proj_t)

        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)

        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        alpha_h_all = self.alpha_embeddings.view(1, -1, self.embedding_dim, 1)
        alpha_rel_h = torch.index_select(self.alpha_left_embeddings, 0, rt_batch[:, 0])
        alpha_rel_h = alpha_rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        alpha_rel_t = torch.index_select(self.alpha_right_embeddings, 0, rt_batch[:, 0])
        alpha_t = torch.index_select(self.alpha_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        beta_h_all = self.beta_embeddings.view(1, -1, self.embedding_dim, 1)
        beta_rel_h = torch.index_select(self.beta_left_embeddings, 0, rt_batch[:, 0])
        beta_rel_h = beta_rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        beta_rel_t = torch.index_select(self.beta_right_embeddings, 0, rt_batch[:, 0])
        beta_t = torch.index_select(self.beta_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            alpha_proj_h = (alpha_rel_h @ alpha_h_all).squeeze(-1)
            alpha_proj_t = (alpha_rel_t @ alpha_t).squeeze(-1)

            beta_proj_h = (beta_rel_h @ beta_h_all).squeeze(-1)
            beta_proj_t = (beta_rel_t @ beta_t).squeeze(-1)

        h_dist = distributions.beta.Beta(alpha_proj_h.unsqueeze(1), beta_proj_h.unsqueeze(1))
        t_dist = distributions.beta.Beta(alpha_proj_t, beta_proj_t)

        scores = -torch.linalg.norm(distributions.kl.kl_divergence(h_dist, t_dist), dim=-1, ord=1)
        return scores

default_biokg_vertex_stalk_dims = {
    'protein':64,
    'function':64,
    'sideeffect':64,
    'drug':64,
    'disease':64
}

default_biokg_edge_stalk_dims = {
    'disease-protein':64,
    'drug-disease':64,
    'drug-drug':64,
    'drug-protein':64,
    'drug-sideeffect':64,
    'function-function':64,
    'protein-function':64,
    'protein-protein':64
}

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

class SheafE_BioKG(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        relation_mapping_location: str,
        edge_stalk_dims: dict = default_biokg_edge_stalk_dims,
        vertex_stalk_dims: dict = default_biokg_vertex_stalk_dims,
        scoring_fct_norm: int = 2,
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
        num_sections: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_constrainer: Optional[Constrainer] = functional.normalize
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.vertex_stalk_dims = vertex_stalk_dims
        self.vertex_class_list = list(self.vertex_stalk_dims.keys())
        self.vertex_to_class_map = self.map_vertices_to_class(triples_factory.entity_to_id, triples_factory.num_entities)
        self.edge_stalk_dims = edge_stalk_dims
        self.edge_class_list = list(self.edge_stalk_dims.keys())
        self.relation_map = self.read_relation_map(relation_mapping_location, triples_factory.relation_to_id)
        self.edge_to_class_map = self.map_edges_to_class()
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.device = preferred_device
        self.entity_constrainer = entity_constrainer

        self.entity_class_embeddings = []
        self.relation_class_embeddings_left = []
        self.relation_class_embeddings_right = []
        self.initialize_entities()
        self.initialize_relations()

    def map_vertices_to_class(self, entity_to_id, num_entities):
        # use the triples factory entity to id map to surmise the entity class.
        # this assumes entity_to_id gives back a dict where the keys are of form
        # entity_class:entity_class_id and the values are their pykeen id.
        # use this pykeen id as the index and use first column as class index,
        # second column as the dimensionality, and third column the index within
        # that class.
        entity_to_id = entity_to_id
        class_count = np.zeros(len(self.vertex_class_list), dtype=np.int)
        retarray = np.empty((num_entities,3), dtype=np.int)
        for k,v in entity_to_id.items():
            name = k[:k.find(':')]
            ix = self.vertex_class_list.index(name)
            retarray[v,0] = ix
            retarray[v,1] = self.vertex_stalk_dims[name]
            retarray[v,2] = class_count[ix]
            class_count[ix] += 1
        return torch.from_numpy(retarray).to(self.device)

    def read_relation_map(self, loc, relation_to_id):
        # load just the column of relation names. each relation id is equal to the index
        rel2id = relation_to_id
        id2rel = {v:int(k) for k,v in rel2id.items()}
        col = np.loadtxt(loc, delimiter=',', skiprows=1, usecols=1, dtype=np.str)
        relations = np.array([col[id2rel[i]] for i in range(col.shape[0])], dtype=np.str)
        return relations

    def map_edges_to_class(self):
        '''Match the edge class prefix by breaking off the part before the underscore
        and then use the index of this class within the edge_class_list as its
        "class" identifier in column 0 and the dimensionality in column 1.
        Column 2 is the count of the number of edges of that particular class that
        has yet been seen. This column is used to index the relation embeddings
        for each class. Columns 3 and 4 are the left and right embedding dimensions,
        respectively.
        '''
        class_count = np.zeros(len(self.edge_class_list))
        retarray = np.empty((self.num_relations, 5), dtype=np.int)
        for rix in range(self.relation_map.shape[0]):
            rname = self.relation_map[rix]
            if '_' in rname:
                rname = rname[:rname.find('_')]
            ix = self.edge_class_list.index(rname)
            ent_left = rname[:rname.find('-')]
            ent_right = rname[rname.find('-')+1:]
            retarray[rix,0] = ix
            retarray[rix,1] = self.edge_stalk_dims[rname]
            retarray[rix,2] = class_count[ix]
            retarray[rix,3] = self.vertex_class_list.index(ent_left)
            retarray[rix,4] = self.vertex_class_list.index(ent_right)
            class_count[ix] += 1
        return torch.from_numpy(retarray).to(self.device)

    def initialize_entities(self):
        for vcix in range(len(self.vertex_class_list)):
            vcname = self.vertex_class_list[vcix]
            nclass = (self.vertex_to_class_map[:,0] == vcix).sum()
            esize = (nclass, self.vertex_stalk_dims[vcname], self.num_sections)
            if self.orthogonal and self.num_sections > 1:
                # is there a faster way to do this? looping over num entities is expensive
                orths = torch.empty(esize, device=self.device, dtype=torch.float32)
                for i in range(esize[0]):
                    orths[i,:,:] = nn.init.orthogonal_(orths[i,:,:])
                self.entity_class_embeddings.append(Parameter(orths, requires_grad=True))
                self.I = torch.eye(self.num_sections, device=self.device)
            else:
                self.entity_class_embeddings.append(Parameter(nn.init.xavier_uniform_(torch.empty(esize, device=self.device, dtype=torch.float32)),requires_grad=True))

    def initialize_relations(self):
        for ecix in range(len(self.edge_class_list)):
            ecname = self.edge_class_list[ecix]
            ent_left = ecname[:ecname.find('-')]
            ent_right = ecname[ecname.find('-')+1:]
            nclass = (self.edge_to_class_map[:,0] == ecix).sum()

            tsizel = (nclass, self.edge_stalk_dims[ecname], self.vertex_stalk_dims[ent_left])
            self.relation_class_embeddings_left.append(Parameter(nn.init.xavier_uniform_(torch.empty(tsizel, device=self.device, dtype=torch.float32)),requires_grad=True))

            tsizer = (nclass, self.edge_stalk_dims[ecname], self.vertex_stalk_dims[ent_right])
            self.relation_class_embeddings_right.append(Parameter(nn.init.xavier_uniform_(torch.empty(tsizer, device=self.device, requires_grad=True, dtype=torch.float32)),requires_grad=True))

    def get_model_savename(self):
        savestruct = 'SheafE_BioKG_{}sec_{}norm'
        if self.orthogonal:
            savestruct += '_{}orthogonal'.format(self.alpha_orthogonal)
        return savestruct.format(self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

    def post_parameter_update(self):
        super().post_parameter_update()
        if self.entity_constrainer is not None:
            for eix in range(len(self.entity_class_embeddings)):
                self.entity_class_embeddings[eix].data = self.entity_constrainer(self.entity_class_embeddings[eix].data, dim=1)

    def get_grad_params(self):
        """Get the parameters that require gradients."""
        return self.relation_class_embeddings_right + self.relation_class_embeddings_left + self.entity_class_embeddings

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        scores = torch.zeros(hrt_batch.shape[0], device=self.device)
        # unique set of relation classes are in this batch.
        rel_classes = torch.unique(torch.index_select(self.edge_to_class_map, 0, hrt_batch[:,1])[:,0])
        # iterate over each relation class and compute the projections, making
        # sure to pick out the entities from their proper class of embeddings
        for i in range(rel_classes.shape[0]):
            cls = rel_classes[i]
            # get indices of all the relations that are in this class
            rel_idxs = torch.squeeze(torch.nonzero(self.edge_to_class_map[:,0] == cls))
            # get the relations in this batch that are within the above set of indices
            # i.e. the relations in this batch that are in class cls
            batch_indices = torch.squeeze(torch.nonzero(isin(hrt_batch[:,1], rel_idxs)))
            batch = hrt_batch[batch_indices]
            # if we have a batch of length 1, this drops a dimension, so unsqueeze
            if len(batch.shape) == 1:
                batch = torch.unsqueeze(batch,0)

            # get a representative head element to infer its class
            # this should be the same across all head elements of this relation class
            h_ent_class = self.vertex_to_class_map[batch[0,0],0].item()
            # get the head dimension of this relation class
            h_ent_class_dim = self.vertex_to_class_map[batch[0,0],1].item()
            h = torch.index_select(self.entity_class_embeddings[h_ent_class], 0, torch.index_select(self.vertex_to_class_map, 0, batch[:,0])[:,2]).view(-1, h_ent_class_dim, self.num_sections)

            # get a representative tail element to infer its class
            # this should be the same across all tail elements of this relation class
            t_ent_class = self.vertex_to_class_map[batch[0,2],0].item()
            # get the tail dimension of this relation class
            t_ent_class_dim = self.vertex_to_class_map[batch[0,2],1].item()
            t = torch.index_select(self.entity_class_embeddings[t_ent_class], 0, torch.index_select(self.vertex_to_class_map, 0, batch[:,2])[:,2]).view(-1, t_ent_class_dim, self.num_sections)

            # select the head and tail projection maps
            rel_h = torch.index_select(self.relation_class_embeddings_left[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,1])[:,2])
            rel_t = torch.index_select(self.relation_class_embeddings_right[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,1])[:,2])

            # project across relations
            proj_h = rel_h @ h
            proj_t = rel_t @ t
            # compute score normally
            score = -torch.norm(proj_h - proj_t, dim=(1,2), p=self.scoring_fct_norm)**2
            if self.orthogonal and self.num_sections > 1:
                ents = torch.cat([h,t],dim=0)
                I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
                I = I.repeat(ents.shape[0], 1, 1)
                orth_score = self.alpha_orthogonal * torch.norm(ents.permute(0,2,1)@ents - I, p=self.scoring_fct_norm)
                score = score - orth_score
            # update score vector at appropriate indices
            scores[batch_indices] = score

        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Need to be careful with tail scoring. For a given hr_batch, we must note
        # the relations composing this batch and only choose tail entities that
        # are of the appropriate class for the tail of that relation.

        # store scores across all classes in this array
        scores = torch.full((hr_batch.shape[0],self.num_entities), float('nan'), device=self.device)
        # unique set of relation classes are in this batch
        rel_classes = torch.unique(torch.index_select(self.edge_to_class_map, 0, hr_batch[:,1])[:,0])
        # iterate over each relation class and compute the projections, making
        # sure to pick out the entities from their proper class of embeddings
        for i in range(rel_classes.shape[0]):
            cls = rel_classes[i]
            # get indices of all the relations that are in this class
            rel_idxs = torch.squeeze(torch.nonzero(self.edge_to_class_map[:,0] == cls))
            # get the relations in this batch that are within the above set of indices
            # i.e. the relations in this batch that are in class cls
            batch_indices = torch.squeeze(torch.nonzero(isin(hr_batch[:,1], rel_idxs)))
            batch = hr_batch[batch_indices]
            # if we have a batch of length 1, this drops a dimension, so unsqueeze
            if len(batch.shape) == 1:
                batch = torch.unsqueeze(batch,0)

            # get a representative head element to infer its class
            # this should be the same across all head elements of this relation class
            h_ent_class = self.vertex_to_class_map[batch[0,0],0].item()
            # get the head dimension of this relation class
            h_ent_class_dim = self.vertex_to_class_map[batch[0,0],1].item()
            h = torch.index_select(self.entity_class_embeddings[h_ent_class], 0, torch.index_select(self.vertex_to_class_map, 0, batch[:,0])[:,2]).view(-1, h_ent_class_dim, self.num_sections)

            # get a representative tail element to infer its class from the info
            # stored in the relation class map. then take all entities in this
            # class as all of the tail entities to compare against
            t_ent_class = self.edge_to_class_map[batch[0,1],4].item()
            # get the tail dimension of this relation class
            t_ent_class_dim = self.vertex_to_class_map[self.vertex_to_class_map[:,0] == t_ent_class][0,1].item()
            t_all = self.entity_class_embeddings[t_ent_class].view(1, -1, t_ent_class_dim, self.num_sections)

            # select the head and tail projection maps
            rel_h = torch.index_select(self.relation_class_embeddings_left[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,1])[:,2])
            rel_t = torch.index_select(self.relation_class_embeddings_right[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,1])[:,2])
            rel_t = rel_t.view(-1, 1, rel_t.shape[-2], rel_t.shape[-1])
            t_indices = torch.nonzero(self.vertex_to_class_map[:,0] == t_ent_class)

            # project across relations
            proj_h = rel_h @ h
            proj_t = rel_t @ t_all
            # compute score normally
            score = -torch.norm(proj_h[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
            # update appropriate indices in the aggregate scores vector
            scores[batch_indices, t_indices] = score.T

        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Need to be careful with head scoring. For a given rt_batch, we must note
        # the relations composing this batch and only choose head entities that
        # are of the appropriate class for the head of that relation.

        # store scores across all classes in this array
        scores = torch.full((rt_batch.shape[0],self.num_entities), float('nan'), device=self.device)
        # unique set of relation classes are in this batch
        rel_classes = torch.unique(torch.index_select(self.edge_to_class_map, 0, rt_batch[:,0])[:,0])
        # iterate over each relation class and compute the projections, making
        # sure to pick out the entities from their proper class of embeddings
        for i in range(rel_classes.shape[0]):
            cls = rel_classes[i]
            # get indices of all the relations that are in this class
            rel_idxs = torch.squeeze(torch.nonzero(self.edge_to_class_map[:,0] == cls))
            # get the relations in this batch that are within the above set of indices
            # i.e. the relations in this batch that are in class cls
            batch_indices = torch.squeeze(torch.nonzero(isin(rt_batch[:,0], rel_idxs)))
            batch = rt_batch[batch_indices]
            # if we have a batch of length 1, this drops a dimension, so unsqueeze
            if len(batch.shape) == 1:
                batch = torch.unsqueeze(batch,0)

            # get a representative tail element to infer its class
            # this should be the same across all tail elements of this relation class
            t_ent_class = self.vertex_to_class_map[batch[0,1],0].item()
            # get the tail dimension of this relation class
            t_ent_class_dim = self.vertex_to_class_map[batch[0,1],1].item()
            t = torch.index_select(self.entity_class_embeddings[t_ent_class], 0, torch.index_select(self.vertex_to_class_map, 0, batch[:,1])[:,2]).view(-1, t_ent_class_dim, self.num_sections)

            # get a representative head element to infer its class
            # this should be the same across all head elements of this relation class
            h_ent_class = self.edge_to_class_map[batch[0,0],3].item()
            # get the head dimension of this relation class
            h_ent_class_dim = self.vertex_to_class_map[self.vertex_to_class_map[:,0] == h_ent_class][0,1].item()
            h_all = self.entity_class_embeddings[h_ent_class].view(1, -1, h_ent_class_dim, self.num_sections)
            h_indices = torch.nonzero(self.vertex_to_class_map[:,0] == h_ent_class)

            # select the head and tail projection maps
            rel_h = torch.index_select(self.relation_class_embeddings_left[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,0])[:,2])
            rel_h = rel_h.view(-1, 1, rel_h.shape[-2], rel_h.shape[-1])
            rel_t = torch.index_select(self.relation_class_embeddings_right[cls], 0, torch.index_select(self.edge_to_class_map, 0, batch[:,0])[:,2])

            # project across relations
            proj_h = rel_h @ h_all
            proj_t = rel_t @ t
            # compute score normally
            score = -torch.norm(proj_h[:, :, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
            # update appropriate indices in the aggregate scores vector
            scores[batch_indices, h_indices] = score.T

        return scores
