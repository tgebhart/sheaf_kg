import os
import sys
import time
import argparse
import json

import pandas as pd
import numpy as np
import pykeen
import torch

from train_sheafE_betae import read_dataset, shuffle_datasets, dataset_to_device, sample_answers
from complex_functions import test_batch
from sheafE_models import SheafE_Multisection, SheafE_Diag, SheafE_Translational, \
                        SheafE_Bilinear, SheafE_Distributional_Normal, SheafE_Distributional_Beta

from pykeen.pipeline import pipeline

dataset = 'FB15k-237'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234
training_loop = 'slcwa'
frequency = 50
patience = 100
num_sections = None
alpha_orthogonal = 0.1
lbda = 0.5
scoring_fct_norm = None
dataset_loc_hint = '../data/{}-betae'

loss = 'SoftplusLoss'

model_map = {'Diag': SheafE_Diag,
            'Multisection': SheafE_Multisection,
            'Translational': SheafE_Translational,
            'Bilinear': SheafE_Bilinear,
            'Distributional_Normal': SheafE_Distributional_Normal,
            'Distributional_Beta': SheafE_Distributional_Beta}

test_query_structures = ['1p','2p','3p','2i','3i','ip','pi']
# test_query_structures = ['pi','ip']

def run(model_name, dataset, num_epochs, embedding_dim, edge_stalk_dim, loss, training_loop, sampler,
    random_seed, num_sections, symmetric, orthogonal, alpha_orthogonal, lbda, scoring_fct_norm,
    model_parameters, model_inverses, test_extension, complex_solver, dataset_loc=None):

    if dataset_loc is None:
        dataset_loc = dataset_loc_hint.format(dataset)

    timestr = time.strftime("%Y%m%d-%H%M")

    model_kwargs = {}
    if model_parameters is not None:
        with open(model_parameters, 'rb') as f:
            model_kwargs = json.load(f)
    # these should be model agnostic
    model_kwargs['embedding_dim'] = embedding_dim
    model_kwargs['symmetric'] = symmetric
    model_kwargs['lbda'] = lbda
    model_kwargs['complex_solver'] = complex_solver
    # these should be model-specific and the args need to be thrown if desired to use.
    if num_sections is not None:
        model_kwargs['num_sections'] = num_sections
    if orthogonal:
        model_kwargs['orthogonal'] = orthogonal
        model_kwargs['alpha_orthogonal'] = alpha_orthogonal
    if edge_stalk_dim is not None:
        model_kwargs['edge_stalk_dim'] = edge_stalk_dim

    if scoring_fct_norm is not None:
        model_kwargs['scoring_fct_norm'] = scoring_fct_norm

    if model_name in model_map:
        model_cls = model_map[model_name]
    else:
        raise ValueError('Model {} not recognized from choices {}'.format(model_name, list(model_map.keys())))

    if dataset == 'NELL995':
        training = os.path.join(dataset_loc, 'train.txt')
        testing = os.path.join(dataset_loc, 'test.txt')
        pykeen_dataset_name = None
    else:
        training = None
        testing = None
        pykeen_dataset_name = dataset

    result = pipeline(
        model=model_cls,
        dataset=pykeen_dataset_name,
        training=training,
        testing=testing,
        random_seed=random_seed,
        device='gpu',
        dataset_kwargs=dict(create_inverse_triples=model_inverses),
        training_kwargs=dict(num_epochs=num_epochs,sampler=sampler),
        evaluation_kwargs=dict(),
        model_kwargs=model_kwargs,
        training_loop=training_loop,
        loss=loss,
        loss_kwargs=dict()
    )

    res_df = result.metric_results.to_df()

    model = result.model
    model_savename = model.get_model_savename()
    savename = model_savename + '_{}epochs_{}loss_{}_{}'.format(num_epochs,loss,sampler,timestr)
    saveloc = os.path.join('../data',dataset,savename)

    if test_extension:
        datasets = read_dataset(dataset_loc, test_query_structures=test_query_structures, model_inverses=model_inverses)
        datasets = dataset_to_device(shuffle_datasets(datasets), model.device)
        extension_df = test_batch(model, datasets['test-easy'], model_inverses=model_inverses, test_query_structures=test_query_structures, complex_solver=complex_solver)
        print(extension_df*100)
        extension_df.to_csv(os.path.join('../data',dataset,'complex',savename+'.csv'))

    res_df.to_csv(saveloc+'.csv')
    result.save_to_directory(saveloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyKeen training run for sheafE models')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310','NELL995'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--edge-stalk-dim', type=int, default=None,
                        help='edge stalk embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=num_sections,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')
    training_args.add_argument('--model', type=str, required=True,
                        choices=['Multisection', 'Diag', 'Translational', 'Bilinear', 'Distributional_Normal', 'Distributional_Beta'],
                        help='name of model to train')
    training_args.add_argument('--training-loop', type=str, required=False, default=training_loop,
                        choices=['slcwa', 'lcwa'],
                        help='closed world assumption')
    training_args.add_argument('--sampler', type=str, required=False, default=None,
                        choices=['schlichtkrull', 'neighborhood', 'geometric'],
                        help='positive batch sampler')
    training_args.add_argument('--symmetric', action='store_true',
                        help='whether to keep restriction maps equivalent on both sides')
    training_args.add_argument('--test-extension', action='store_true',
                        help='whether to test harmonic extension on complex queries')
    training_args.add_argument('--orthogonal', action='store_true',
                        help='whether to learn orthogonalization of each entity vector (section)')
    training_args.add_argument('--alpha-orthogonal', type=float, default=alpha_orthogonal,
                        help='hyperparameter weighting on orthogonal term of scoring')
    training_args.add_argument('--lbda', type=float, default=lbda,
                        help='hyperparameter weighting on batch-wise scoring for neighborhood sampling')
    training_args.add_argument('--scoring-fct-norm', type=int, default=scoring_fct_norm,
                        help='scoring norm to use')
    training_args.add_argument('--model-parameters', type=str, required=False, default=None,
                        help='path to json file of model-specific parameters')
    training_args.add_argument('--model-inverses', action='store_true',
                        help='whether to explicitly model inverse relations')
    training_args.add_argument('--complex-solver', type=str, required=False, default='schur',
                        help='which complex queries harmonic extension solver to use')
    training_args.add_argument('--dataset-loc', type=str, default=None,
                        help='full path location to betae remapped dataset')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.embedding_dim, args.edge_stalk_dim, args.loss,
        args.training_loop, args.sampler, args.seed, args.num_sections, args.symmetric,
        args.orthogonal, args.alpha_orthogonal, args.lbda, args.scoring_fct_norm, args.model_parameters,
        args.model_inverses, args.test_extension, args.complex_solver)
