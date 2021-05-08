import os
import sys
import time
import argparse
import pickle
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
import pykeen
from pykeen.losses import SoftplusLoss, CrossEntropyLoss, BCEWithLogitsLoss
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.triples import CoreTriplesFactory
import torch
from torch_geometric.data import NeighborSampler, Data, ShaDowKHopSampler, \
                                GraphSAINTRandomWalkSampler, RandomNodeSampler, \
                                ClusterData, ClusterLoader

from sheafE_models import SheafE_Multisection, SheafE_Translational
from sheaf_kg.complex_functions import test_batch as test
from pykeen.pipeline import pipeline

dataset = 'FB15k-237'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234
training_loop = 'slcwa'
alpha_orthogonal = 0.1
test_every = 1
scoring_fct_norm = 2
batch_size = 100
test_batch_size = 5
dataset_loc = '/home/gebhart/projects/sheaf_kg/data/{}-betae'.format(dataset)
loss = 'SoftplusLoss'

train_query_structures = ['1p','2p','3p','2i','3i']
test_query_structures = ['1p','2p','3p','2i','3i','ip','pi']

model_map = {'Diagonal': SheafE_Translational,
            'Multisection': SheafE_Multisection}

def read_dataset(loc, train_query_structures=train_query_structures, test_query_structures=test_query_structures, model_inverses=False):
    dsets = {'train':{}, 'test-easy':{}}
    resaved_remapped = 'resaved' if model_inverses else 'remapped'
    for key_name in dsets.keys():
        if key_name == 'train':
            query_structures = train_query_structures
        else:
            query_structures = test_query_structures
        for query_structure in query_structures:
            entities = np.load(os.path.join(dataset_loc, f'{query_structure}_{resaved_remapped}_{key_name}_entities.npy'))
            relations = np.load(os.path.join(dataset_loc, f'{query_structure}_{resaved_remapped}_{key_name}_relations.npy'))
            inverses = np.load(os.path.join(dataset_loc, f'{query_structure}_{resaved_remapped}_{key_name}_inverses.npy'))
            with open(os.path.join(dataset_loc, f'{query_structure}_{resaved_remapped}_{key_name}_answers.pkl'), 'rb') as f:
                answers = pickle.load(f)
            dsets[key_name][query_structure] = {}
            dsets[key_name][query_structure]['entities'] = entities
            dsets[key_name][query_structure]['relations'] = relations
            dsets[key_name][query_structure]['inverses'] = inverses
            dsets[key_name][query_structure]['answers'] = answers
    return dsets

def shuffle_datasets(dsets):
    # this ignores the answers key, as it is assumed these will be dealt with separately
    for key_name in dsets:
        dset = dsets[key_name]
        for query_structure in dset:
            n_queries = len(dset[query_structure]['answers'])
            p = np.random.permutation(n_queries)
            for obj in dset[query_structure]:
                if obj != 'answers':
                    dsets[key_name][query_structure][obj] = dsets[key_name][query_structure][obj][p]
                else:
                    ans = dsets[key_name][query_structure][obj]
                    dsets[key_name][query_structure][obj] = [ans[i] for i in p]
    return dsets

def dataset_to_device(dsets, device):
    # this ignores the answers key, as it is assumed these will be dealt with separately
    for key_name in dsets:
        dset = dsets[key_name]
        for query_structure in dset:
            for obj in dset[query_structure]:
                if obj != 'answers':
                    dsets[key_name][query_structure][obj] = torch.LongTensor(dset[query_structure][obj]).to(device)
    return dsets

def sample_answers(answers):
    return torch.LongTensor([np.random.choice(answer_list) for answer_list in answers])

def run(model_name, dataset, dataset_loc, num_epochs, batch_size, test_every, embedding_dim, edge_stalk_dim, loss_name, training_loop,
    random_seed, num_sections, symmetric, orthogonal, alpha_orthogonal, scoring_fct_norm, model_inverses):

    # torch.autograd.set_detect_anomaly(True)

    timestr = time.strftime("%Y%m%d-%H%M")
    ds = pykeen.datasets.get_dataset(dataset=dataset, dataset_kwargs=dict(create_inverse_triples=model_inverses))
    num_entities = ds.num_entities
    num_relations = ds.num_relations

    datasets = read_dataset(dataset_loc)

    device = 'cuda'

    model_kwargs = {}
    model_kwargs['embedding_dim'] = embedding_dim
    if edge_stalk_dim is not None:
        model_kwargs['edge_stalk_dim'] = edge_stalk_dim
    model_kwargs['num_sections'] = num_sections
    model_kwargs['symmetric'] = symmetric
    model_kwargs['orthogonal'] = orthogonal
    model_kwargs['alpha_orthogonal'] = alpha_orthogonal
    model_kwargs['scoring_fct_norm'] = scoring_fct_norm
    model_kwargs['preferred_device'] = device

    model_kwargs['restrict_identity'] = True

    if model_name in model_map:
        model_cls = model_map[model_name]
    else:
        raise ValueError('Model {} not recognized from choices {}'.format(model_name, list(model_map.keys())))

    model = model_cls(num_entities, num_relations, **model_kwargs)
    datasets = dataset_to_device(shuffle_datasets(datasets), device)
    ds = pykeen.datasets.get_dataset(dataset=dataset, dataset_kwargs=dict(create_inverse_triples=model_inverses))
    training = ds.training.mapped_triples

    edge_index = torch.empty((2,training.shape[0]), dtype=torch.int64)
    edge_index[0,:] = training[:,0]
    edge_index[1,:] = training[:,2]
    train_idx = torch.unique(edge_index.flatten())
    # https://docs.dgl.ai/en/0.4.x/tutorials/models/5_giant_graph/1_sampling_mx.html
    train_loader = NeighborSampler(edge_index, node_idx=train_idx,
                               sizes=[1,1,1,1,1,1], batch_size=10,
                               shuffle=True, num_workers=2)

    tf = CoreTriplesFactory(training, train_idx.shape[0], torch.unique(training[:,1]).shape[0], create_inverse_triples=model_inverses)
    neg_sampler = BernoulliNegativeSampler(tf)

    optimizer = torch.optim.Adam(
            model.get_parameters(),
            lr=1e-4)
    gamma = 0.1
    step_size = 2
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)

    if loss_name == 'SoftplusLoss':
        loss_function = SoftplusLoss()
    if loss_name == 'CrossEntropyLoss':
        loss_function = CrossEntropyLoss()
    if loss_name == 'BCEWithLogitsLoss':
        loss_function = BCEWithLogitsLoss()

    results = []
    for epoch in range(1,num_epochs+1):
        # model.train()
        print(f'epoch: {epoch}')

        for batch_size, n_id, adjs in tqdm(train_loader):

            optimizer.zero_grad()
            loss = 0

            e_ids = []
            for j, (eix, e_id, size) in enumerate(adjs):
                e_ids.append(e_id)

            pos_batch = training[torch.cat(e_ids, dim=0)].to(device)
            neg_batch = neg_sampler.sample(pos_batch)[0]

            pos_answers = torch.ones(pos_batch.shape[0]).to(device)
            neg_answers = torch.zeros(neg_batch.shape[0]).to(device)

            pos_scores = model.score_hrt(pos_batch)
            neg_scores = model.score_hrt(neg_batch)

            scores = torch.cat([pos_scores,neg_scores], dim=0)
            answers = torch.cat([pos_answers, neg_answers], dim=0)
            loss += loss_function(scores, answers)

            loss.backward()
            optimizer.step()
            model.post_parameter_update()

        print('Train Loss: {:.2f}'.format(loss.item()))

        if epoch % test_every == 0:
            res_df = test(model, datasets['test-easy'], model_inverses=model_inverses, test_query_structures=test_query_structures)
            res_df['epoch'] = epoch
            results.append(res_df)
            print(res_df*100)

        # scheduler.step()

    res_df = pd.concat(results, axis=0)


    # model = result.model
    model_savename = model.get_model_savename()
    savename = model_savename + '_{}epochs_{}_{}'.format(num_epochs,loss_name,timestr)
    saveloc = os.path.join('../data',dataset,savename)

    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    res_df.to_csv(os.path.join(saveloc, 'test_results.csv'))
    torch.save(model, os.path.join(saveloc, 'trained_model.pkl'))
    # result.save_to_directory(saveloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyKeen training run for sheafE models')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='FB15k-237',
                        choices=['WN18RR','WN18','FB15k','FB15k-237','YAGO310'],
                        help='dataset (default: WN18RR)')
    training_args.add_argument('--model', type=str, required=True,
                        choices=['Multisection', 'Diagonal'],
                        help='name of model to train')
    training_args.add_argument('--dataset-loc', type=str, default=dataset_loc,
                        help='full path location to betae remapped dataset')
    training_args.add_argument('--num-epochs', type=int, default=num_epochs,
                        help='number of training epochs')
    training_args.add_argument('--batch-size', type=int, default=batch_size,
                        help='training/testing batch size')
    training_args.add_argument('--test-every', type=int, default=test_every,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=embedding_dim,
                        help='entity embedding dimension')
    training_args.add_argument('--edge-stalk-dim', type=int, default=None,
                        help='entity embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=1,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--seed', type=int, default=random_seed,
                        help='random seed')
    training_args.add_argument('--loss', type=str, default=loss,
                        help='loss function')
    training_args.add_argument('--training-loop', type=str, required=False, default=training_loop,
                        choices=['slcwa', 'lcwa'],
                        help='closed world assumption')
    training_args.add_argument('--symmetric', action='store_true',
                        help='whether to keep restriction maps equivalent on both sides')
    training_args.add_argument('--orthogonal', action='store_true',
                        help='whether to learn orthogonalization of each entity vector (section)')
    training_args.add_argument('--alpha-orthogonal', type=float, default=alpha_orthogonal,
                        help='hyperparameter weighting on orthogonal term of scoring')
    training_args.add_argument('--scoring-fct-norm', type=int, default=scoring_fct_norm,
                        help='scoring norm to use')
    training_args.add_argument('--model-inverses', action='store_true',
                        help='whether to explicitly model inverse relations')

    args = parser.parse_args()

    run(args.model, args.dataset, args.dataset_loc, args.num_epochs, args.batch_size, args.test_every,
        args.embedding_dim, args.edge_stalk_dim, args.loss, args.training_loop,
        args.seed, args.num_sections, args.symmetric, args.orthogonal,
        args.alpha_orthogonal, args.scoring_fct_norm, args.model_inverses)
