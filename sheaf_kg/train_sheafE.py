import os
import sys
import time
import argparse
import json

import pandas as pd
import numpy as np
import pykeen
import torch

from sheafE_models import SheafE_Multisection, SheafE_Diag, SheafE_Translational

from pykeen.pipeline import pipeline

dataset = 'WN18RR'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234
training_loop = 'slcwa'
frequency = 50
patience = 100
alpha_orthogonal = 0.1

loss = 'SoftplusLoss'

model_map = {'Diag': SheafE_Diag,
            'Multisection': SheafE_Multisection,
            'Translational': SheafE_Translational}

def run(model_name, dataset, num_epochs, embedding_dim, loss, training_loop,
    random_seed, num_sections, symmetric, orthogonal, alpha_orthogonal, model_parameters):

    timestr = time.strftime("%Y%m%d-%H%M")

    model_kwargs = {}
    if model_parameters is not None:
        with open(model_parameters, 'rb') as f:
            model_kwargs = json.load(f)
    model_kwargs['embedding_dim'] = embedding_dim
    model_kwargs['num_sections'] = num_sections
    model_kwargs['symmetric'] = symmetric
    model_kwargs['orthogonal'] = orthogonal
    model_kwargs['alpha_orthogonal'] = alpha_orthogonal

    if model_name in model_map:
        model_cls = model_map[model_name]
    else:
        raise ValueError('Model {} not recognized from choices {}'.format(model_name, list(model_map.keys())))

    result = pipeline(
        model=model_cls,
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=model_kwargs,
        # stopper='early',
        training_loop=training_loop,
        # stopper_kwargs=dict(frequency=frequency, patience=patience),
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    model = result.model
    model_savename = model.get_model_savename()
    savename = model_savename + '_{}epochs_{}loss_{}'.format(num_epochs,loss,timestr)
    saveloc = os.path.join('../data',dataset,savename)

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyKeen training run for sheafE models')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=1,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')
    training_args.add_argument('--model', type=str, required=True,
                        choices=['Multisection', 'Diag', 'Translational'],
                        help='name of model to train')
    training_args.add_argument('--training-loop', type=str, required=False, default=training_loop,
                        choices=['slcwa', 'lcwa'],
                        help='closed world assumption')
    training_args.add_argument('--symmetric', action='store_true',
                        help='whether to keep restriction maps equivalent on both sides')
    training_args.add_argument('--orthogonal', action='store_true',
                        help='whether to learn orthogonalization of each entity vector (section)')
    training_args.add_argument('--alpha-orthogonal', type=float, default=alpha_orthogonal,
                        help='hyperparameter weighting on orthogonal term of scoring')
    training_args.add_argument('--model-parameters', type=str, required=False, default=None,
                        help='path to json file of model-specific parameters')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.embedding_dim, args.loss,
        args.training_loop, args.seed, args.num_sections, args.symmetric,
        args.orthogonal, args.alpha_orthogonal, args.model_parameters)
