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
from pykeen.models.base import _OldAbstractModel
from pykeen.nn import Embedding
from pykeen.losses import Loss
from pykeen.nn.init import xavier_uniform_
from pykeen.regularizers import Regularizer
from pykeen.triples import TriplesFactory
from pykeen.typing import DeviceHint
from pykeen.utils import compose

from torch.nn import functional
from torch import nn

dataset = 'WN18RR'
num_epochs = 1000
embedding_dim = 64
num_sections = 50
random_seed = 1234

loss = 'SoftplusLoss'
timestr = time.strftime("%Y%m%d-%H%M")

class ModifiedSE(_OldAbstractModel):

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 64,
        scoring_fct_norm: int = 2,
        num_sections: int = 50,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for SE.
        """
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.embedding_dim = embedding_dim
        self.num_sections = num_sections
        self.scoring_fct_norm = scoring_fct_norm

        esize = (triples_factory.num_entities, num_sections, embedding_dim)
        self.ent_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(esize, device=preferred_device, dtype=torch.float32)),requires_grad=True)

        tsize = (triples_factory.num_relations, embedding_dim, embedding_dim)
        self.left_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, dtype=torch.float32)),requires_grad=True)
        self.right_embeddings = Parameter(nn.init.xavier_uniform_(torch.empty(tsize, device=preferred_device, requires_grad=True, dtype=torch.float32)),requires_grad=True)

    def _reset_parameters_(self):  # noqa: D102
        self.ent_embeddings = nn.init.xavier_uniform_(self.ent_embeddings)
        self.left_embeddings = nn.init.xavier_uniform_(self.left_embeddings)
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
        rel_t = rel_t.view(-1, 1, self.embedding_dim, self.embedding_dim)
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
        rel_h = rel_h.view(-1, 1, self.embedding_dim, self.embedding_dim)
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


def run(dataset, num_epochs, embedding_dim, loss, num_sections, random_seed):

    savename = 'SheafE_multisection_{}num_sections_{}epochs_{}dim_{}loss_{}seed_{}'.format(num_sections,num_epochs,embedding_dim,loss,random_seed,timestr)
    saveloc = os.path.join('../data',dataset,savename)

    result = pipeline(
        model=ModifiedSE,
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        stopper='early',
        stopper_kwargs=dict(frequency=50, patience=100),
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=dict(embedding_dim=embedding_dim, num_sections=num_sections),
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)

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
    training_args.add_argument('--num-sections', type=int, default=num_sections,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')

    args = parser.parse_args()

    run(args.dataset, args.num_epochs, args.embedding_dim, args.loss, args.num_sections, args.seed)
