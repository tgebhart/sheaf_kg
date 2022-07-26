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
import torch

from sheaf_kg.sheafE_models import SheafE_Multisection, SheafE_Translational

from pykeen.pipeline import pipeline

dataset = 'FB15k-237'
num_epochs = 1000
embedding_dim = 64
random_seed = 1234
training_loop = 'slcwa'
alpha_orthogonal = 0.1
scoring_fct_norm = 2
batch_size = 100
test_batch_size = 5
dataset_loc = '../data/{}-betae'.format(dataset)
loss = 'MarginRankingLoss'

# train_query_structures = ['1p','2p','3p','2i','3i']
train_query_structures = ['1p','2p','3p']
# train_query_structures = ['2i','3i']
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
            entities = np.load(os.path.join(loc, f'{query_structure}_{resaved_remapped}_{key_name}_entities.npy'))
            relations = np.load(os.path.join(loc, f'{query_structure}_{resaved_remapped}_{key_name}_relations.npy'))
            inverses = np.load(os.path.join(loc, f'{query_structure}_{resaved_remapped}_{key_name}_inverses.npy'))
            with open(os.path.join(loc, f'{query_structure}_{resaved_remapped}_{key_name}_answers.pkl'), 'rb') as f:
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

def test(model, test_data, model_inverses=False, sec=0, test_batch_size=test_batch_size):
    with torch.no_grad():
        allhits1 = []
        allhits3 = []
        allhits5 = []
        allhits10 = []
        allmrr = []
        query_names = []
        for query_structure in test_query_structures:
            print('Running query : {}'.format(query_structure))
            hits1 = 0.
            hits3 = 0.
            hits5 = 0.
            hits10 = 0.
            mrr = 0.
            cnt = 0
            num_test = len(test_data[query_structure]['answers'])
            for qix in tqdm(range(0, num_test//2, test_batch_size)):
                if num_test - qix == 1:
                    continue
                entities = test_data[query_structure]['entities'][qix:qix+test_batch_size]
                relations = test_data[query_structure]['relations'][qix:qix+test_batch_size]
                if model_inverses:
                    inverses = None
                else:
                    inverses = test_data[query_structure]['inverses'][qix:qix+test_batch_size]
                all_answers = test_data[query_structure]['answers'][qix:qix+test_batch_size]
                targets = torch.arange(model.num_entities).to(model.device)
                Q = model.forward_costs(query_structure, entities, relations, targets, invs=inverses)
                Q = Q[:,:,sec]
                for i in range(Q.shape[0]):
                    Qi = Q[i].squeeze()
                    answers = all_answers[i]
                    sortd,_ = torch.sort(Qi)
                    idxleft = torch.searchsorted(sortd, Qi[answers], right=False) + 1
                    idxright = torch.searchsorted(sortd, Qi[answers], right=True) + 1
                    nl = idxleft.shape[0]
                    nr = idxright.shape[0]
                    # idxright = idxleft # throw this for optimistic ranking
                    hits1 += ((torch.sum(idxleft <= 1)/nl + torch.sum(idxright <= 1)/nr) / 2.)
                    hits3 += ((torch.sum(idxleft <= 3)/nl + torch.sum(idxright <= 3)/nr) / 2.)
                    hits5 += ((torch.sum(idxleft <= 5)/nl + torch.sum(idxright <= 5)/nr) / 2.)
                    hits10 += ((torch.sum(idxleft <= 10)/nl + torch.sum(idxright <= 10)/nr) / 2.)
                    mrr += ((torch.sum(1./idxleft)/nl + torch.sum(1./idxright)/nr) / 2.)
                    cnt += 1
            if cnt > 0:
                allhits1.append(hits1.item()/cnt)
                allhits3.append(hits3.item()/cnt)
                allhits5.append(hits5.item()/cnt)
                allhits10.append(hits10.item()/cnt)
                allmrr.append(mrr.item()/cnt)
            else:
                default = 0.
                allhits1.append(default)
                allhits3.append(default)
                allhits5.append(default)
                allhits10.append(default)
                allmrr.append(default)

        cols = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'mrr']
        df = pd.DataFrame(np.array([allhits1, allhits3, allhits5, allhits10, allmrr]).T, columns=cols, index=test_query_structures)
        return df

def run(model_name, dataset, dataset_loc, num_epochs, batch_size, embedding_dim, edge_stalk_dim, loss_name, training_loop,
    random_seed, num_sections, symmetric, orthogonal, alpha_orthogonal, scoring_fct_norm, model_inverses):

    torch.autograd.set_detect_anomaly(True)

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
    train_data = datasets['train']

    optimizer = torch.optim.Adam(
            model.get_parameters(),
            lr=1e-2)
    gamma = 0.1
    step_size = 2
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)

    if loss_name == 'SoftplusLoss':
        loss_function = SoftplusLoss()
    if loss_name == 'CrossEntropyLoss':
        loss_function = CrossEntropyLoss()
    if loss_name == 'BCEWithLogitsLoss':
        loss_function = BCEWithLogitsLoss()

    # all train query structures should have the same number of queries
    num_queries = train_data[train_query_structures[0]]['relations'].shape[0]

    results = []

    for epoch in range(1,num_epochs+1):
        # model.train()
        print(f'epoch: {epoch}')

        for bix in tqdm(range(0, num_queries, batch_size)):

            optimizer.zero_grad()
            loss = 0
            for query_structure in train_data:

                entities = train_data[query_structure]['entities'][bix:bix+batch_size]
                relations = train_data[query_structure]['relations'][bix:bix+batch_size]
                targets = train_data[query_structure]['answers'][bix:bix+batch_size]
                if model_inverses:
                    inverses = None
                else:
                    inverses = train_data[query_structure]['inverses'][bix:bix+batch_size]

                sampled_targets = sample_answers(targets).to(device)
                sampled_answers = torch.ones(sampled_targets.shape).to(device)
                neg_targets = torch.randint(model.num_entities, sampled_targets.shape).to(device)
                neg_answers = torch.zeros(neg_targets.shape).to(device)
                scores_sampled = model.score_query(query_structure, entities, relations, sampled_targets, invs=inverses)
                scores_neg = model.score_query(query_structure, entities, relations, neg_targets, invs=inverses)
                scores = torch.cat([scores_sampled,scores_neg], dim=0)
                answers = torch.cat([sampled_answers, neg_answers], dim=0)
                loss += loss_function(scores, answers)

            loss.backward()
            optimizer.step()
            model.post_parameter_update()

        print('Train Loss: {:.2f}'.format(loss.item()))

        res_df = test(model, datasets['test-easy'])
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

    run(args.model, args.dataset, args.dataset_loc, args.num_epochs, args.batch_size,
        args.embedding_dim, args.edge_stalk_dim, args.loss, args.training_loop,
        args.seed, args.num_sections, args.symmetric, args.orthogonal,
        args.alpha_orthogonal, args.scoring_fct_norm, args.model_inverses)
