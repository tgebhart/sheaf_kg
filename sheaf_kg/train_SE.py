import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pykeen
import torch

from pykeen.pipeline import pipeline

dataset = 'WN18RR'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234

loss = 'SoftplusLoss'
timestr = time.strftime("%Y%m%d-%H%M")

def run(dataset, num_epochs, embedding_dim, loss, random_seed):

    savename = 'SE_{}epochs_{}dim_{}loss_{}seed_{}'.format(num_epochs,embedding_dim,loss,random_seed,timestr)
    saveloc = os.path.join('/home/gebhart/projects/sheaf_kg/data',dataset,savename)

    result = pipeline(
        model='StructuredEmbedding',
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=dict(embedding_dim=embedding_dim),
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SE training run')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--random-seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')

    args = parser.parse_args()

    run(args.dataset, args.num_epochs, args.embedding_dim, args.loss, args.random_seed)
