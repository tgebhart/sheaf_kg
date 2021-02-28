import os
import sys
import time
import argparse
import json

import pandas as pd
import numpy as np
import pykeen
import torch

from sheafE_models import SheafE_BioKG

from pykeen.pipeline import pipeline
from pykeen.sampling import EntityTypeSampler

dataset = 'OGBBioKG'
num_epochs = 1000
random_seed = 1234
training_loop = 'slcwa'
frequency = 50
patience = 100
alpha_orthogonal = 0.01
relation_mapping_location = '/home/gebhart/.data/pykeen/datasets/ogbbiokg/ogbl_biokg/mapping/relidx2relname.csv.gz'
biokg_type_list = ['protein','function','sideeffect','drug','disease']

loss = 'SoftplusLoss'

def run(num_epochs, loss, training_loop, random_seed, num_sections, orthogonal, alpha_orthogonal, model_parameters):

    timestr = time.strftime("%Y%m%d-%H%M")

    model_kwargs = {}
    if model_parameters is not None:
        with open(model_parameters, 'rb') as f:
            model_kwargs = json.load(f)
    model_kwargs['num_sections'] = num_sections
    model_kwargs['orthogonal'] = orthogonal
    model_kwargs['alpha_orthogonal'] = alpha_orthogonal
    model_kwargs['relation_mapping_location'] = relation_mapping_location

    result = pipeline(
        model=SheafE_BioKG,
        dataset=dataset,
        random_seed=random_seed,
        device='gpu',
        training_kwargs=dict(num_epochs=num_epochs),
        evaluation_kwargs=dict(),
        model_kwargs=model_kwargs,
        stopper='early',
        training_loop=training_loop,
        stopper_kwargs=dict(frequency=frequency, patience=patience),
        loss=loss,
        loss_kwargs=dict(),
        negative_sampler=EntityTypeSampler,
        negative_sampler_kwargs=dict(entity_types=biokg_type_list)
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
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--num-sections', type=int, default=1,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')
    training_args.add_argument('--training-loop', type=str, required=False, default=training_loop,
                        choices=['slcwa', 'lcwa'],
                        help='closed world assumption')
    training_args.add_argument('--orthogonal', action='store_true',
                        help='whether to learn orthogonalization of each entity vector (section)')
    training_args.add_argument('--alpha-orthogonal', type=float, default=alpha_orthogonal,
                        help='hyperparameter weighting on orthogonal term of scoring')
    training_args.add_argument('--model-parameters', type=str, required=False, default=None,
                        help='path to json file of model-specific parameters')

    args = parser.parse_args()

    run(args.num_epochs, args.loss, args.training_loop, args.seed, args.num_sections,
        args.orthogonal, args.alpha_orthogonal, args.model_parameters)
