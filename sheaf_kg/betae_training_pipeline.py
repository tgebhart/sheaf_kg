import os
import argparse

import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.typing import LABEL_TAIL
from sheaf_kg.models.betae_extension_structured_embedding import BetaeExtensionStructuredEmbedding
from sheaf_kg.models.betae_extension_translational import BetaeExtensionTranslational, BetaeExtensionTransE
from sheaf_kg.regularizers.multisection_regularizers import OrthogonalSectionsRegularizer
from sheaf_kg.evaluation.evaluator import BetaeEvaluator
from sheaf_kg.data_loader import generate_mapped_triples

DATASET = 'FB15k-237'
BASE_DATA_PATH = 'KG_data'
MODEL = 'BetaeExtensionStructuredEmbedding'
PARAMETERIZATION = None
NUM_EPOCHS = 1
C0_DIM = 32
C1_DIM = 32
NUM_SECTIONS = 1
RANDOM_SEED = 134
TRAINING_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 32
REGULARIZER_WEIGHT = 0.1
QUERY_STRUCTURES  = ['1p','2p','3p','2i','3i','pi','ip']
SAMPLED_ANSWERS = True
SLICE_SIZE = None
TEST_TYPES = ['easy','hard']

model_map = {
    'BetaeExtensionTranslational':BetaeExtensionTranslational,
    'BetaeExtensionStructuredEmbedding':BetaeExtensionStructuredEmbedding,
    'BetaeExtensionTransE':BetaeExtensionTransE,
    'BetaeNaiveTransE':BetaeExtensionTransE
}

def find_dataset_betae(dataset):
    return {'train': f'{BASE_DATA_PATH}/{dataset}-betae/train.txt', 
            'validate': f'{BASE_DATA_PATH}/{dataset}-betae/valid.txt', 
            'test': {
                'easy': {
                    'answers': f'{BASE_DATA_PATH}/{dataset}-betae/test-easy-answers.pkl', 
                    'queries': f'{BASE_DATA_PATH}/{dataset}-betae/test-queries.pkl',
                },
                'hard': {
                    'answers': f'{BASE_DATA_PATH}/{dataset}-betae/test-hard-answers.pkl', 
                    'queries': f'{BASE_DATA_PATH}/{dataset}-betae/test-queries.pkl',
                },
                
                'triples': f'{BASE_DATA_PATH}/{dataset}-betae/test.txt'},
            'ent2id': f'{BASE_DATA_PATH}/{dataset}-betae/ent2id.pkl',
            'rel2id': f'{BASE_DATA_PATH}/{dataset}-betae/rel2id.pkl'
            }

def find_model(model):
    if model in model_map:
        return model_map[model]
    raise ValueError(f'model {model} not known')

def find_parameterization(param):
    if param == 'orthogonal':
        return torch.nn.utils.parametrizations.orthogonal
    return None

def get_factories(dataset, query_structures=QUERY_STRUCTURES, 
                sampled_answers=SAMPLED_ANSWERS, test_types=TEST_TYPES):

    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset)
    train = triples_factory.from_path(dstf['train'], create_inverse_triples=False)
    # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
    test = triples_factory.from_path(dstf['test']['triples'], create_inverse_triples=False, 
                                    entity_to_id=train.entity_to_id, relation_to_id=train.relation_to_id)

    # we will also need to remap the test complex queries
    e2id = {int(k):v for k,v in train.entity_to_id.items()}
    r2id = {int(k):v for k,v in train.relation_to_id.items()}
    def remap_fun(q, ans):
        q['relations'] = q['relations'].apply_(r2id.get)
        q['sources'] = q['sources'].apply_(e2id.get)
        ans = [e2id[a] for a in ans]
        return q, ans
    
    test_queries = {}
    for test_type in test_types:
        test_queries[test_type] = generate_mapped_triples(dstf['test'][test_type]['queries'], dstf['test'][test_type]['answers'], 
                                        random_sample=sampled_answers, query_structures=query_structures, 
                                        remap_fun=remap_fun)
    return train, test, test_queries

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, num_sections=NUM_SECTIONS, reg_weight=REGULARIZER_WEIGHT, 
        parameterization=PARAMETERIZATION, 
        training_batch_size=TRAINING_BATCH_SIZE, evaluation_batch_size=EVALUATION_BATCH_SIZE, slice_size=SLICE_SIZE,
        query_structures=QUERY_STRUCTURES, sampled_answers=SAMPLED_ANSWERS, test_types=TEST_TYPES):

    train_tf, test_tf, test_queries = get_factories(dataset, query_structures=query_structures, 
                                                    sampled_answers=sampled_answers, test_types=test_types)
    model_class = find_model(model)
    parameterization_fun = find_parameterization(parameterization)
    evaluator = BetaeEvaluator(filtered=sampled_answers)
    
    model_kwargs = {}
    model_kwargs['num_sections'] = num_sections
    
    if model == 'BetaeExtensionTransE':
        model_kwargs['embedding_dim'] = embedding_dim
    elif model == 'BetaeNaiveTransE':
        model_kwargs['embedding_dim'] = embedding_dim
        model_kwargs['naive_extension'] = True
    else:
        model_kwargs['restriction_parameterization'] = parameterization_fun
        model_kwargs['C0_dimension'] = embedding_dim
        if c1_dimension is not None:
            model_kwargs['C1_dimension'] = c1_dimension
        
    train_device = 'cuda'
    evaluate_device = 'cuda'

    result = pipeline(
        model=model_class,
        model_kwargs=model_kwargs,
        regularizer=OrthogonalSectionsRegularizer,
        regularizer_kwargs={"weight": reg_weight},
        training=train_tf,
        testing=test_tf,
        training_kwargs={'batch_size':training_batch_size},
        epochs=num_epochs,
        device=train_device,
        random_seed=random_seed,
    )

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])
    mr['query_structure'] = 'test'

    # save out
    savedir = f'data/{dataset}/BetaE/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_sections}sec_{parameterization}param_{reg_weight}reg_{num_epochs}epochs'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if parameterization is not None:
        torch.save(result.model.state_dict(), os.path.join(savedir, 'model.pt'))
    else:
        result.save_to_directory(savedir)

    savename = 'metric_results_{}.csv'
    for test_type in test_types:
        rdfs = [mr]
        for query_structure in query_structures:
            print(f'evaluating {test_type} query structure {query_structure}')
            results = evaluator.evaluate(
                device=evaluate_device,
                model=result.model,
                mapped_triples=test_queries[test_type][query_structure],
                targets=[LABEL_TAIL],
                batch_size=evaluation_batch_size,
                slice_size=slice_size
            )

            ev_df = results.to_df()
            ev_df['query_structure'] = query_structure
            rdfs.append(ev_df)
        rdfs = pd.concat(rdfs, axis=0)
        print(rdfs)
        
        rdfs.to_csv(os.path.join(savedir, savename.format(test_type)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--num-epochs', type=int, default=NUM_EPOCHS,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=C0_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--c1-dimension', type=int, default=C1_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=NUM_SECTIONS,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='random seed')
    training_args.add_argument('--model', type=str, required=False, default=MODEL,
                        choices=list(model_map.keys()),
                        help='name of model to train')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, num_sections=args.num_sections)


