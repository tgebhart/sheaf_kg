import os
import sys
import time
import argparse

import pandas as pd
import numpy as np
import pykeen
import torch

from pykeen.pipeline import pipeline

dataset = 'WN18RR'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234
training_loop = 'slcwa'
dataset_loc_hint = '../data/{}-betae'

loss = 'SoftplusLoss'

def run(model, dataset, num_epochs, embedding_dim, loss, training_loop, random_seed, dataset_loc=None):

    timestr = time.strftime("%Y%m%d-%H%M")
    savename = '{}_{}epochs_{}embdim_{}loss_{}seed_{}'.format(model, num_epochs,embedding_dim,loss,random_seed,timestr)
    saveloc = os.path.join('../data',dataset,savename)

    if dataset_loc is None:
        dataset_loc = dataset_loc_hint.format(dataset)

    if dataset == 'NELL995':
        training = os.path.join(dataset_loc, 'train.txt')
        testing = os.path.join(dataset_loc, 'test.txt')
        pykeen_dataset_name = None
    else:
        training = None
        testing = None
        pykeen_dataset_name = dataset

    result = pipeline(
        model=model,
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=dict(embedding_dim=embedding_dim),
        stopper='early',
        training_loop=training_loop,
        stopper_kwargs=dict(frequency=50, patience=100),
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyKeen training run')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310', 'OGBBioKG'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')
    training_args.add_argument('--model', type=str, required=True,
                        help='Pykeen name of model to train')
    training_args.add_argument('--training-loop', type=str, required=False, default=training_loop,
                        choices=['slcwa', 'lcwa'],
                        help='closed world assumption')
    training_args.add_argument('--dataset-loc', type=str, default=None,
                        help='full path location to the custom dataset')


    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.embedding_dim, args.loss,
        args.training_loop, args.seed, dataset_loc = args.dataset_loc)
