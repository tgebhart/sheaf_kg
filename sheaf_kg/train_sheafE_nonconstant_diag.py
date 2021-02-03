import os
import sys
import time
import argparse
import functools
import itertools
from typing import Optional

import pandas as pd
import numpy as np
import pykeen
import torch

from pykeen.pipeline import pipeline

from pykeen.models import StructuredEmbedding
from pykeen.models.base import EntityEmbeddingModel
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

dataset = 'WN18RR'
num_epochs = 1000
embedding_dim = 64
edge_stalk_sizes = [1,1,1,1,1,1,1,1,1,1,1]
random_seed = 1234

loss = 'SoftplusLoss'
timestr = time.strftime("%Y%m%d-%H%M")

class ModifiedSE(StructuredEmbedding):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        edge_stalk_sizes: [int] = [1,1,1,1,1,1,1,1,1,1,1],
        alpha: float = 0.1,
        scoring_fct_norm: int = 2,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.alpha = alpha
        self.preferred_device = preferred_device

        # Embeddings
        init_bound = 6 / np.sqrt(self.embedding_dim)

        self.left_embeddings = []
        self.right_embeddings = []

        # relation 1
        tsize = (edge_stalk_sizes[0],embedding_dim)
        emb1l = Parameter(nn.init.xavier_uniform_(torch.eye(tsize[0],tsize[1], device=preferred_device)),requires_grad=True)
        emb1r = emb1l
        self.left_embeddings.append(emb1l)
        self.right_embeddings.append(emb1r)

        # relation 2
        tsize = (edge_stalk_sizes[1],embedding_dim)
        emb2l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb2r = emb2l
        self.left_embeddings.append(emb2l)
        self.right_embeddings.append(emb2r)

        # relation 3
        tsize = (edge_stalk_sizes[2],embedding_dim)
        emb3l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)), requires_grad=True)
        emb3r = emb3l
        self.left_embeddings.append(emb3l)
        self.right_embeddings.append(emb3r)

        # relation 4
        tsize = (edge_stalk_sizes[3],embedding_dim)
        emb4l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)), requires_grad=True)
        emb4r = emb4l
        self.left_embeddings.append(emb4l)
        self.right_embeddings.append(emb4r)

        # relation 5
        tsize = (edge_stalk_sizes[4],embedding_dim)
        emb5l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb5r = emb5l
        self.left_embeddings.append(emb5l)
        self.right_embeddings.append(emb5r)

        # relation 6
        tsize = (edge_stalk_sizes[5],embedding_dim)
        emb6l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb6r = emb6l
        self.left_embeddings.append(emb6l)
        self.right_embeddings.append(emb6r)

        # relation 7
        tsize = (edge_stalk_sizes[6],embedding_dim)
        emb7l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb7r = emb7l
        self.left_embeddings.append(emb7l)
        self.right_embeddings.append(emb7r)

        # relation 8
        tsize = (edge_stalk_sizes[7],embedding_dim)
        emb8l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb8r = emb8l
        self.left_embeddings.append(emb8l)
        self.right_embeddings.append(emb8r)

        # relation 9
        tsize = (edge_stalk_sizes[8],embedding_dim)
        emb9l = Parameter(nn.init.xavier_uniform_(torch.eye(tsize[0],tsize[1], device=preferred_device)),requires_grad=True)
        emb9r = emb9l
        self.left_embeddings.append(emb9l)
        self.right_embeddings.append(emb9r)

        # relation 10
        tsize = (edge_stalk_sizes[9],embedding_dim)
        emb10l = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device)),requires_grad=True)
        emb10r = emb10l
        self.left_embeddings.append(emb10l)
        self.right_embeddings.append(emb10r)

        # relation 11
        tsize = (edge_stalk_sizes[10],embedding_dim)
        emb11l = Parameter(nn.init.xavier_uniform_(torch.eye(tsize[0],tsize[1], device=preferred_device)),requires_grad=True)
        emb11r = emb11l
        self.left_embeddings.append(emb11l)
        self.right_embeddings.append(emb11r)


    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        rel_idxs = torch.unique(hrt_batch[:,1])
        scores = torch.zeros(hrt_batch.shape[0], device=self.preferred_device)
        for i in range(rel_idxs.shape[0]):
            ix = rel_idxs[i]
            batch_indices = torch.nonzero(hrt_batch[:,1] == ix)
            batch = hrt_batch[hrt_batch[:,1] == ix]
            h = self.entity_embeddings(indices=batch[:, 0]).view(-1, self.embedding_dim, 1)
            t = self.entity_embeddings(indices=batch[:, 2]).view(-1, self.embedding_dim, 1)
            rel_h = self.left_embeddings[ix]
            rel_t = self.right_embeddings[ix]

            proj_h = torch.diagflat(rel_h) @ h
            proj_t = torch.diagflat(rel_t) @ t
            scores[batch_indices] = -torch.norm(proj_h - proj_t, dim=1, p=self.scoring_fct_norm)**2

        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        rel_idxs = torch.unique(hr_batch[:,1])
        t_all = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim, 1)
        scores = torch.zeros((hr_batch.shape[0],t_all.shape[1]), device=self.preferred_device)
        for i in range(rel_idxs.shape[0]):
            ix = rel_idxs[i]
            batch_indices = torch.nonzero(hr_batch[:,1] == ix)
            batch = hr_batch[hr_batch[:,1] == ix]
            h = self.entity_embeddings(indices=batch[:, 0]).view(-1, self.embedding_dim, 1)
            rel_h = self.left_embeddings[ix]
            rel_t = self.right_embeddings[ix]
            rel_t = rel_t.view(-1, 1, rel_t.shape[0], rel_t.shape[1])

            if slice_size is not None:
                proj_t_arr = []
                # Project entities
                proj_h = rel_h @ h

                for t in torch.split(t_all, slice_size, dim=1):
                    # Project entities
                    proj_t = rel_t @ t
                    proj_t_arr.append(proj_t)

                proj_t = torch.cat(proj_t_arr, dim=1)

            else:
                proj_h = torch.diagflat(rel_h) @ h
                proj_t = torch.diagflat(rel_t) @ t_all

            scores[batch_indices[:,0]] = -torch.norm(proj_h[:, None, :, 0] - proj_t[:, :, :, 0], dim=-1, p=self.scoring_fct_norm)**2

        return scores


    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        rel_idxs = torch.unique(rt_batch[:,0])
        h_all = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim, 1)
        scores = torch.zeros((rt_batch.shape[0],h_all.shape[1]), device=self.preferred_device)
        for i in range(rel_idxs.shape[0]):
            ix = rel_idxs[i]
            batch_indices = torch.nonzero(rt_batch[:,0] == ix)
            batch = rt_batch[rt_batch[:,0] == ix]
            t = self.entity_embeddings(indices=batch[:, 1]).view(-1, self.embedding_dim, 1)
            rel_h = self.left_embeddings[ix]
            rel_h = rel_h.view(-1, 1, rel_h.shape[0], rel_h.shape[1])
            rel_t = self.right_embeddings[ix]

            if slice_size is not None:
                proj_t_arr = []
                # Project entities
                proj_h = rel_h @ h

                for t in torch.split(t_all, slice_size, dim=1):
                    # Project entities
                    proj_t = rel_t @ t
                    proj_t_arr.append(proj_t)

                proj_t = torch.cat(proj_t_arr, dim=1)

            else:
                proj_h = torch.diagflat(rel_h) @ h_all
                proj_t = torch.diagflat(rel_t) @ t

            scores[batch_indices[:,0]] = -torch.norm(proj_h[:, :, :, 0] - proj_t[:, None, :, 0], dim=-1, p=self.scoring_fct_norm)**2

        return scores


def run(dataset, num_epochs, embedding_dim, loss, random_seed):

    savename = 'SheafE_nonconstant_diag_{}epochs_{}dim_{}loss_{}seed_{}'.format(num_epochs,embedding_dim,loss,random_seed,timestr)
    saveloc = os.path.join('../data',dataset,savename)
    esss = np.array(edge_stalk_sizes, dtype='int')

    result = pipeline(
        model=ModifiedSE,
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        stopper='early',
        stopper_kwargs=dict(frequency=50, patience=100),
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=dict(embedding_dim=embedding_dim, edge_stalk_sizes=edge_stalk_sizes),
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)
    np.savetxt(os.path.join(saveloc, 'edge_stalk_sizes.csv'), esss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SheafE - nonconstant training run')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')

    args = parser.parse_args()

    run(args.dataset, args.num_epochs, args.embedding_dim, args.loss, args.seed)
