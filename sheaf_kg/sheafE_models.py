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
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
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

        self.symmetric = bool(symmetric)
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.device = preferred_device

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
            savestruct = 'SheafE_Multisection_Symmetric_{}embdim_{}esdim_{}sec_{}norm'
        else:
            savestruct = 'SheafE_Multisection_{}embdim_{}esdim_{}sec_{}norm'
        if self.orthogonal:
            savestruct += '_{}orthogonal'.format(self.alpha_orthogonal)
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

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
        if self.orthogonal and self.num_sections > 1:
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            I = I.repeat(ents.shape[0], 1, 1)
            orth_scores = self.alpha_orthogonal * torch.norm(ents.permute(0,2,1)@ents - I, p=self.scoring_fct_norm)
            return scores - orth_scores
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
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
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
        self.symmetric = bool(symmetric)
        self.embedding_dim = embedding_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.device = preferred_device

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
        tsize = (self.num_relations, self.embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)),requires_grad=True)
        if self.symmetric:
            self.right_embeddings = self.left_embeddings
        else:
            self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def get_model_savename(self):
        if self.symmetric:
            savestruct = 'SheafE_Diag_Symmetric_{}embdim_{}sec_{}norm'
        else:
            savestruct = 'SheafE_Diag_{}embdim_{}sec_{}norm'
        return savestruct.format(self.embedding_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):  # noqa: D102
        self.initialize_entities()
        self.initialize_relations()

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
        if self.orthogonal and self.num_sections > 1:
            nh = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
            nt = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2]).view(-1, self.embedding_dim, self.num_sections)
            ents = torch.cat([nh,nt],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            I = I.repeat(ents.shape[0], 1, 1)
            orth_scores = self.alpha_orthogonal * torch.norm(ents.permute(0,2,1)@ents - I, p=self.scoring_fct_norm)
            return scores - orth_scores
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





class SheafE_Translational(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = True, # this will be true by definition
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
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
        self.symmetric = True
        self.embedding_dim = embedding_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.device = preferred_device

        self.initialize_entities()
        self.initialize_relations()
        self.initialize_edge_cochains()

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
        id = torch.eye(self.edge_stalk_dim, self.embedding_dim, device=self.device, dtype=torch.float32)
        id = id.reshape((1, self.edge_stalk_dim, self.embedding_dim))
        batch_id = id.repeat(self.num_relations, 1, 1)
        self.left_embeddings = batch_id
        self.right_embeddings = batch_id

    def initialize_edge_cochains(self):
        tsize = (self.num_relations, self.edge_stalk_dim)
        self.edge_cochains = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)

    def get_model_savename(self):
        savestruct = 'SheafE_Translational_{}embdim_{}esdim_{}sec_{}norm'
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):
        self.initialize_entities()
        self.initialize_relations()
        self.initialize_edge_cochains()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0])
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1])
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1])
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2])
        c = torch.index_select(self.edge_cochains, 0, hrt_batch[:,1]).view(-1, self.edge_stalk_dim, 1)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t
        scores = -torch.norm(proj_h + c - proj_t, dim=(1,2), p=self.scoring_fct_norm)
        if self.orthogonal and self.num_sections > 1:
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            I = I.repeat(ents.shape[0], 1, 1)
            orth_scores = self.alpha_orthogonal * torch.norm(ents.permute(0,2,1)@ents - I, p=self.scoring_fct_norm)
            return scores - orth_scores
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hr_batch[:, 1])
        c = torch.index_select(self.edge_cochains, 0, hr_batch[:,1]).view(-1, self.edge_stalk_dim, 1)
        rel_t = torch.index_select(self.right_embeddings, 0, hr_batch[:, 1])
        rel_t = rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        t_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            proj_h = rel_h @ h
            proj_t = rel_t @ t_all

        scores = -torch.norm(proj_h[:, None, :, :] + c[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, rt_batch[:, 0])
        rel_h = rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        c = torch.index_select(self.edge_cochains, 0, rt_batch[:,0]).view(-1, self.edge_stalk_dim, 1)
        rel_t = torch.index_select(self.right_embeddings, 0, rt_batch[:, 0])
        t = torch.index_select(self.ent_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            proj_h = rel_h @ h_all
            proj_t = rel_t @ t

        scores = -torch.norm(proj_h[:, :, :, :] + c[:, None, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores



class SheafE_Bilinear(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_dim: int = 64,
        scoring_fct_norm: int = 2,
        symmetric: bool = True,
        orthogonal: bool = False,
        alpha_orthogonal: float = 0.1,
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
        self.orthogonal = bool(orthogonal)
        self.alpha_orthogonal = alpha_orthogonal
        self.device = preferred_device

        self.initialize_entities()
        self.initialize_relations()
        self.initialize_edge_cochains()

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

    def initialize_edge_cochains(self):
        tsize = (self.num_relations, self.edge_stalk_dim)
        self.edge_cochains = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=self.device, dtype=torch.float32)), requires_grad=True)

    def get_model_savename(self):
        savestruct = 'SheafE_Bilinear_{}embdim_{}esdim_{}sec_{}norm'
        return savestruct.format(self.embedding_dim, self.edge_stalk_dim, self.num_sections, self.scoring_fct_norm)

    def _reset_parameters_(self):
        self.initialize_entities()
        self.initialize_relations()
        self.initialize_edge_cochains()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 0])
        rel_h = torch.index_select(self.left_embeddings, 0, hrt_batch[:, 1])
        rel_t = torch.index_select(self.right_embeddings, 0, hrt_batch[:, 1])
        t = torch.index_select(self.ent_embeddings, 0, hrt_batch[:, 2])
        c = torch.index_select(self.edge_cochains, 0, hrt_batch[:,1]).view(-1, self.edge_stalk_dim, 1)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t
        scores = -torch.norm(proj_h + c - proj_t, dim=(1,2), p=self.scoring_fct_norm)
        if self.orthogonal and self.num_sections > 1:
            ents = torch.cat([h,t],dim=0)
            I = self.I.reshape((1, self.I.shape[0], self.I.shape[1]))
            I = I.repeat(ents.shape[0], 1, 1)
            orth_scores = self.alpha_orthogonal * torch.norm(ents.permute(0,2,1)@ents - I, p=self.scoring_fct_norm)
            return scores - orth_scores
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = torch.index_select(self.ent_embeddings, 0, hr_batch[:, 0]).view(-1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, hr_batch[:, 1])
        c = torch.index_select(self.edge_cochains, 0, hr_batch[:,1]).view(-1, self.edge_stalk_dim, 1)
        rel_t = torch.index_select(self.right_embeddings, 0, hr_batch[:, 1])
        rel_t = rel_t.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        t_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')

        else:
            # Project entities
            proj_h = rel_h @ h
            proj_t = rel_t @ t_all

        scores = -torch.norm(proj_h[:, None, :, :] + c[:, None, :, :] - proj_t[:, :, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h_all = self.ent_embeddings.view(1, -1, self.embedding_dim, self.num_sections)
        rel_h = torch.index_select(self.left_embeddings, 0, rt_batch[:, 0])
        rel_h = rel_h.view(-1, 1, self.edge_stalk_dim, self.embedding_dim)
        c = torch.index_select(self.edge_cochains, 0, rt_batch[:,0]).view(-1, self.edge_stalk_dim, 1)
        rel_t = torch.index_select(self.right_embeddings, 0, rt_batch[:, 0])
        t = torch.index_select(self.ent_embeddings, 0, rt_batch[:, 1]).view(-1, self.embedding_dim, self.num_sections)

        if slice_size is not None:
            raise ValueError('Not implemented')
        else:
            # Project entities
            proj_h = rel_h @ h_all
            proj_t = rel_t @ t

        scores = -torch.norm(proj_h[:, :, :, :] + c[:, None, :, :] - proj_t[:, None, :, :], dim=(-1,-2), p=self.scoring_fct_norm)
        return scores
