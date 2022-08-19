import argparse

import matplotlib.pyplot as plt
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import LABEL_TAIL
from sheaf_kg.models.extension_structured_embedding import ExtensionStructuredEmbedding
from sheaf_kg.models.betae_extension_structured_embedding import BetaeExtensionStructuredEmbedding
from sheaf_kg.regularizers.multisection_regularizers import OrthogonalSectionsRegularizer
from sheaf_kg.models.multisection_structured_embedding import MultisectionStructuredEmbedding
from sheaf_kg.models.multisection_trans_e import MultisectionTransE
from sheaf_kg.evaluation.evaluator import BetaeEvaluator
from sheaf_kg.data_loader import generate_mapped_triples

DATASET = 'FB15k-237'
BASE_DATA_PATH = 'KG_data'
MODEL = 'BetaeExtensionStructuredEmbedding'
SAMPLER = 'schlichtkrull'
NUM_EPOCHS = 25
C0_DIM = 50
C1_DIM = 20
NUM_SECTIONS = 1
RANDOM_SEED = 134
REGULARIZER_WEIGHT = 1
QUERY_STRUCTURES  = ['1p','2p','3p','2i','3i','pi','ip']

model_map = {
    'MultisectionStructuredEmbedding':MultisectionStructuredEmbedding,
    'MultisectionTransE':MultisectionTransE,
    'ExtensionStructuredEmbedding':ExtensionStructuredEmbedding,
    'BetaeExtensionStructuredEmbedding':BetaeExtensionStructuredEmbedding
}

def find_dataset(dataset):
    return {'train': f'{BASE_DATA_PATH}/{dataset}-betae/train.txt', 
            'validate': f'{BASE_DATA_PATH}/{dataset}-betae/valid.txt', 
            'test': {
                'answers': f'{BASE_DATA_PATH}/{dataset}-betae/test-easy-answers.pkl', 
                'queries': f'{BASE_DATA_PATH}/{dataset}-betae/test-queries.pkl'}
            }

def find_model(model):
    if model in model_map:
        return model_map[model]
    raise ValueError(f'model {model} not known')

def get_factories(dataset):

    triples_factory = TriplesFactory

    dstf = find_dataset(dataset)
    train = triples_factory.from_path(dstf['train'])
    validate = triples_factory.from_path(dstf['validate'])
    test_queries = generate_mapped_triples(dstf['test']['queries'], dstf['test']['answers'], random_sample=False)
    return train, validate, test_queries

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, num_sections=None,
        sampler=None, query_structures = QUERY_STRUCTURES):

    
    train_tf, validate_tf, test_queries = get_factories(dataset)
    model_class = find_model(model)
    reg_weight = REGULARIZER_WEIGHT
        
    evaluator = BetaeEvaluator(filtered=True)
    
    model_kwargs = {}
    model_kwargs['C0_dimension'] = embedding_dim
    if num_sections is not None:
        model_kwargs['num_sections'] = num_sections
    if c1_dimension is not None:
        model_kwargs['C1_dimension'] = c1_dimension 
    
    # model_kwargs['training_mask_pct'] = 0.1
    train_device = 'cuda'
    evaluate_device = 'cuda'
    model_inst = model_class(triples_factory=train_tf, **model_kwargs).to(train_device)

    tl = SLCWATrainingLoop(model=model_inst, 
                        triples_factory=train_tf)
    tl.train(train_tf, 
            num_epochs=num_epochs)

    rdfs = []
    for query_structure in query_structures:
        print(f'evaluating query structure {query_structure}')
        results = evaluator.evaluate(
            device=evaluate_device,
            model=tl.model,
            mapped_triples=test_queries[query_structure],
            targets=[LABEL_TAIL],
        )

        ev_df = results.to_df()
        ev_df['query_structure'] = query_structure
        rdfs.append(ev_df)
    rdfs = pd.concat(rdfs, axis=0)
    print(rdfs)
    rdfs.to_csv(f'data/{dataset}/BetaE/{model}_{reg_weight}reg_{num_epochs}epochs.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        choices=['WN18RR','FB15k-237', 'Nations'],
                        help='dataset (default: Nations)')
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
                        choices=['MultisectionStructuredEmbedding', 'MultisectionTransE'],
                        help='name of model to train')
    training_args.add_argument('--sampler', type=str, required=False, default=SAMPLER,
                        choices=['linear', 'complex', 'schlichtkrull'],
                        help='name of triples factory to use')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, num_sections=args.num_sections,
        sampler=args.sampler)


